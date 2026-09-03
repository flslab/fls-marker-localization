#include "localization_pipeline.h"
#include "pose_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class TestFailure : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void check(bool condition, const char *expression, const char *file, int line) {
  if (condition) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": check failed: " << expression;
  throw TestFailure(message.str());
}

void checkNear(double actual, double expected, double tolerance,
               const char *actual_expression, const char *expected_expression,
               const char *file, int line) {
  if (std::isfinite(actual) && std::isfinite(expected) &&
      std::abs(actual - expected) <= tolerance) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": expected " << actual_expression
          << " ~= " << expected_expression << " (actual " << actual
          << ", expected " << expected << ", tolerance " << tolerance << ')';
  throw TestFailure(message.str());
}

template <typename Callable>
void checkThrowsContaining(Callable &&callable, const std::string &needle,
                           const char *file, int line) {
  try {
    callable();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(needle) != std::string::npos) {
      return;
    }
    std::ostringstream message;
    message << file << ':' << line << ": exception did not contain '" << needle
            << "': " << error.what();
    throw TestFailure(message.str());
  }
  std::ostringstream message;
  message << file << ':' << line << ": expected exception containing '"
          << needle << "'";
  throw TestFailure(message.str());
}

#define CHECK(expression)                                                      \
  check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance)                                \
  checkNear((actual), (expected), (tolerance), #actual, #expected, __FILE__,   \
            __LINE__)
#define CHECK_THROWS_CONTAINING(expression, needle)                            \
  checkThrowsContaining([&]() { expression; }, (needle), __FILE__, __LINE__)

std::string fixturePath(const std::string &filename) {
  return std::string(TEST_FIXTURE_DIR) + '/' + filename;
}

RelativeMarker observation(std::size_t detection_index, int id, int row,
                           int col, bool accepted = true) {
  RelativeMarker marker;
  marker.detection_index = detection_index;
  marker.image_x = 100.0F + static_cast<float>(col);
  marker.image_y = 200.0F + static_cast<float>(row);
  marker.id = id;
  marker.row = row;
  marker.col = col;
  marker.accepted = accepted;
  return marker;
}

std::vector<RelativeMarker>
windowObservations(const MarkerGrid &grid, int map_row, int map_col,
                   int window_size, int relative_row, int relative_col) {
  std::vector<RelativeMarker> result;
  for (int dr = 0; dr < window_size; ++dr) {
    for (int dc = 0; dc < window_size; ++dc) {
      const std::size_t index = result.size();
      result.push_back(observation(index,
                                   grid.cells()[map_row + dr][map_col + dc],
                                   relative_row + dr, relative_col + dc));
    }
  }
  return result;
}

template <typename Value>
std::vector<Value> reordered(const std::vector<Value> &values,
                             const std::vector<std::size_t> &order) {
  CHECK(values.size() == order.size());
  std::vector<Value> result;
  result.reserve(values.size());
  for (const std::size_t index : order) {
    CHECK(index < values.size());
    result.push_back(values[index]);
  }
  return result;
}

cv::Mat cameraMatrix(double fx = 800.0, double fy = 800.0, double cx = 320.0,
                     double cy = 240.0) {
  return cv::Mat(cv::Matx33d(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)).clone();
}

cv::Matx33d rpyRotation(const cv::Vec3d &degrees) {
  constexpr double pi = 3.14159265358979323846;
  const double roll = degrees[0] * pi / 180.0;
  const double pitch = degrees[1] * pi / 180.0;
  const double yaw = degrees[2] * pi / 180.0;
  const cv::Matx33d rx(1.0, 0.0, 0.0, 0.0, std::cos(roll), -std::sin(roll), 0.0,
                       std::sin(roll), std::cos(roll));
  const cv::Matx33d ry(std::cos(pitch), 0.0, std::sin(pitch), 0.0, 1.0, 0.0,
                       -std::sin(pitch), 0.0, std::cos(pitch));
  const cv::Matx33d rz(std::cos(yaw), -std::sin(yaw), 0.0, std::sin(yaw),
                       std::cos(yaw), 0.0, 0.0, 0.0, 1.0);
  return rz * ry * rx;
}

cv::Vec4d quaternionFromRpyDegrees(const cv::Vec3d &degrees) {
  constexpr double pi = 3.14159265358979323846;
  const double half_roll = degrees[0] * pi / 360.0;
  const double half_pitch = degrees[1] * pi / 360.0;
  const double half_yaw = degrees[2] * pi / 360.0;
  const double cr = std::cos(half_roll);
  const double sr = std::sin(half_roll);
  const double cp = std::cos(half_pitch);
  const double sp = std::sin(half_pitch);
  const double cy = std::cos(half_yaw);
  const double sy = std::sin(half_yaw);
  return {sr * cp * cy - cr * sp * sy,
          cr * sp * cy + sr * cp * sy,
          cr * cp * sy - sr * sp * cy,
          cr * cp * cy + sr * sp * sy};
}

cv::Point2f projectPlanePoint(double x, double y, double distance,
                              const cv::Matx33d &plane_to_camera,
                              const cv::Mat &intrinsics) {
  const cv::Vec3d camera_point = plane_to_camera * cv::Vec3d(x, y, distance);
  return {static_cast<float>(intrinsics.at<double>(0, 0) * camera_point[0] /
                                 camera_point[2] +
                             intrinsics.at<double>(0, 2)),
          static_cast<float>(intrinsics.at<double>(1, 1) * camera_point[1] /
                                 camera_point[2] +
                             intrinsics.at<double>(1, 2))};
}

MarkerDetection gridCoordinateDetection(
    double row_coordinate, double col_coordinate, int id,
    std::uint64_t track_id, double spacing, double distance,
    const cv::Matx33d &grid_to_camera_rotation, const cv::Mat &intrinsics) {
  const cv::Point2f image =
      projectPlanePoint(-row_coordinate * spacing,
                        -col_coordinate * spacing, distance,
                        grid_to_camera_rotation, intrinsics);
  MarkerDetection detection;
  detection.x = image.x;
  detection.y = image.y;
  detection.id = id;
  detection.track_id = track_id;
  return detection;
}

MarkerDetection mapCellDetection(
    const MarkerGrid &grid, int row, int col, std::uint64_t track_id,
    double row_offset, double col_offset, double distance,
    const cv::Matx33d &grid_to_camera_rotation, const cv::Mat &intrinsics,
    bool decoded) {
  return gridCoordinateDetection(
      row + row_offset, col + col_offset,
      decoded ? grid.cells()[row][col] : -1, track_id, grid.cellSpacing(),
      distance, grid_to_camera_rotation, intrinsics);
}

MarkerDetection worldPointDetection(
    const cv::Point3f &point, int id, std::uint64_t track_id,
    const cv::Point2d &camera_foot, double distance,
    const cv::Matx33d &grid_to_camera_rotation, const cv::Mat &intrinsics,
    bool decoded) {
  const cv::Point2f image = projectPlanePoint(
      point.x - camera_foot.x, point.y - camera_foot.y, distance,
      grid_to_camera_rotation, intrinsics);
  MarkerDetection detection;
  detection.x = image.x;
  detection.y = image.y;
  detection.id = decoded ? id : -1;
  detection.track_id = track_id;
  return detection;
}

const MarkerDetection *findTrack(const std::vector<MarkerDetection> &detections,
                                 std::uint64_t track_id) {
  const auto found = std::find_if(
      detections.begin(), detections.end(),
      [track_id](const MarkerDetection &detection) {
        return detection.track_id == track_id;
      });
  return found == detections.end() ? nullptr : &*found;
}

void testQuaternionFrameMath() {
  const cv::Matx33d &camera_to_drone = pose_math::cameraToDroneRotation();
  CHECK_NEAR(cv::norm(camera_to_drone * cv::Vec3d(1.0, 0.0, 0.0) -
                      cv::Vec3d(0.0, -1.0, 0.0)),
             0.0, 1e-12);
  CHECK_NEAR(cv::norm(camera_to_drone * cv::Vec3d(0.0, 1.0, 0.0) -
                      cv::Vec3d(-1.0, 0.0, 0.0)),
             0.0, 1e-12);
  CHECK_NEAR(cv::norm(camera_to_drone * cv::Vec3d(0.0, 0.0, 1.0) -
                      cv::Vec3d(0.0, 0.0, -1.0)),
             0.0, 1e-12);
  CHECK_NEAR(cv::determinant(cv::Mat(camera_to_drone)), 1.0, 1e-12);

  const auto identity =
      pose_math::gridToCameraRotationFromDroneQuaternionXyzw(
          {0.0, 0.0, 0.0, 3.0});
  CHECK(identity.has_value());
  CHECK_NEAR(cv::norm(cv::Mat(*identity - camera_to_drone.t())), 0.0, 1e-12);

  const double half_sqrt = std::sqrt(0.5);
  const cv::Vec4d yaw_quaternion(0.0, 0.0, half_sqrt, half_sqrt);
  const auto yaw_rotation =
      pose_math::gridToCameraRotationFromDroneQuaternionXyzw(yaw_quaternion);
  const auto negated_rotation =
      pose_math::gridToCameraRotationFromDroneQuaternionXyzw(-yaw_quaternion);
  CHECK(yaw_rotation.has_value());
  CHECK(negated_rotation.has_value());
  const cv::Matx33d expected_yaw(1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                                 0.0, 0.0, -1.0);
  CHECK_NEAR(cv::norm(cv::Mat(*yaw_rotation - expected_yaw)), 0.0, 1e-12);
  CHECK_NEAR(cv::norm(cv::Mat(*yaw_rotation - *negated_rotation)), 0.0,
             1e-12);

  const cv::Vec4d tilted_quaternion =
      quaternionFromRpyDegrees({8.0, -12.0, 17.0});
  const auto drone_to_world =
      pose_math::rotationFromQuaternionXyzw(tilted_quaternion);
  const auto world_to_camera =
      pose_math::gridToCameraRotationFromDroneQuaternionXyzw(
          tilted_quaternion);
  CHECK(drone_to_world.has_value());
  CHECK(world_to_camera.has_value());
  const cv::Vec3d expected_drone_position(0.4, -0.7, 2.2);
  const cv::Vec3d camera_offset_drone(0.015, -0.035, -0.035);
  const cv::Vec3d expected_camera_position =
      expected_drone_position + *drone_to_world * camera_offset_drone;
  const cv::Vec3d pnp_translation =
      -(*world_to_camera * expected_camera_position);
  const cv::Vec3d recovered_camera_position =
      -(world_to_camera->t() * pnp_translation);
  const cv::Vec3d recovered_drone_position =
      recovered_camera_position - *drone_to_world * camera_offset_drone;
  CHECK_NEAR(cv::norm(recovered_camera_position - expected_camera_position),
             0.0, 1e-12);
  CHECK_NEAR(cv::norm(recovered_drone_position - expected_drone_position), 0.0,
             1e-12);

  CHECK(pose_math::normalizeQuaternionXyzw({1e-300, 0.0, 0.0, 0.0})
            .has_value());
  CHECK(!pose_math::normalizeQuaternionXyzw({0.0, 0.0, 0.0, 0.0})
             .has_value());
  CHECK(!pose_math::normalizeQuaternionXyzw(
             {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0})
             .has_value());
}

void testMapParsingAndValidation() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  CHECK(grid.rows() == 5);
  CHECK(grid.cols() == 5);
  CHECK(grid.numIds() == 18);
  CHECK(grid.minK() == 3);
  CHECK(grid.windowSize() == 2);
  CHECK(grid.totalWindowCount() == 16);
  CHECK(grid.uniqueWindowCount() == 16);
  CHECK(grid.maxWindowOccurrences() == 1);
  CHECK_NEAR(grid.cellSpacing(), 0.25, 1e-7);

  const cv::Point3f point = grid.cellToGlobal(2, 3);
  CHECK_NEAR(point.x, 0.5, 1e-7);
  CHECK_NEAR(point.y, -2.75, 1e-7);
  CHECK_NEAR(point.z, 0.5, 1e-7);

  CHECK_THROWS_CONTAINING(grid.cellToGlobal(-1, 0), "out of bounds");
  CHECK_THROWS_CONTAINING(
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 1), "window_size");
  CHECK_THROWS_CONTAINING(
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 6), "window_size");
  CHECK_THROWS_CONTAINING(
      MarkerGrid::fromJson(fixturePath("invalid_grid.json"), 2), "rectangular");
  CHECK_THROWS_CONTAINING(
      MarkerGrid::fromJson(fixturePath("does_not_exist.json"), 2),
      "failed to open");
}

void testShortRangeMapParsingAndValidation() {
  const std::string path = fixturePath("short_range_grid.json");
  const MarkerGrid main = MarkerGrid::fromJson(path, 2);
  const ShortRangeMarkerGrid short_range =
      ShortRangeMarkerGrid::fromJson(path, main);
  CHECK(main.workingRange().has_value());
  CHECK_NEAR(main.workingRange()->min_distance, 0.3, 1e-12);
  CHECK_NEAR(main.workingRange()->max_distance, 2.0, 1e-12);
  CHECK(short_range.enabled());
  CHECK(short_range.windowSize() == 2);
  CHECK_NEAR(short_range.cellSpacing(), 0.02, 1e-7);
  CHECK_NEAR(short_range.markerSize(), 0.006, 1e-7);
  CHECK(short_range.workingRange().has_value());
  CHECK_NEAR(short_range.workingRange()->min_distance, 0.05, 1e-12);
  CHECK_NEAR(short_range.workingRange()->max_distance, 0.5, 1e-12);
  CHECK(short_range.tiles().size() == 2);
  CHECK(short_range.markerIds() == std::set<int>({16, 17}));
  CHECK(short_range.tiles()[0].i == 0);
  CHECK(short_range.tiles()[0].j == 0);
  CHECK(short_range.tiles()[0].signature ==
        std::vector<int>({16, 16, 16, 16}));
  CHECK_NEAR(short_range.tiles()[0].markers[3].global_position.x, 0.09,
             1e-7);
  CHECK_NEAR(short_range.tiles()[0].markers[3].global_position.y, 0.09,
             1e-7);

  CHECK(main.containsWindowSignature({0, 1, 4, 5}));
  CHECK(!main.containsWindowSignature({16, 16, 16, 16}));
  CHECK_THROWS_CONTAINING(MarkerGrid::fromJson(path, 3), "--window-size");

  // Old maps without declared ranges or a root window size remain valid, and
  // signature lookup still uses the signature's side length.
  const MarkerGrid legacy_main = MarkerGrid::fromJson(
      fixturePath("short_range_overlap_grid.json"), 3);
  CHECK(!legacy_main.workingRange().has_value());
  CHECK(legacy_main.containsWindowSignature({0, 1, 3, 4}));
  const MarkerGrid old_main =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  CHECK(!ShortRangeMarkerGrid::fromJson(fixturePath("unique_grid.json"),
                                        old_main)
             .enabled());
  CHECK_THROWS_CONTAINING(
      ShortRangeMarkerGrid::fromJson(
          fixturePath("short_range_overlap_grid.json"),
          MarkerGrid::fromJson(fixturePath("short_range_overlap_grid.json"),
                               2)),
      "disjoint");
  CHECK_THROWS_CONTAINING(
      MarkerGrid::fromJson(fixturePath("invalid_working_range_grid.json"), 2),
      "0 <= min_distance <= max_distance");
}

void testDecoderGate() {
  const std::set<std::uint64_t> seen = {1, 2, 3, 4};
  const std::set<std::uint64_t> known_main = {1, 2};
  const std::set<std::uint64_t> known_short = {4, 8};
  CHECK(decoderIgnoredTracksForGridSelection(
            false, false, seen, known_main, known_short) == known_short);
  CHECK(decoderIgnoredTracksForGridSelection(
            false, true, seen, known_main, known_short) ==
        std::set<std::uint64_t>({3, 4, 8}));
  CHECK(decoderIgnoredTracksForGridSelection(
            true, false, seen, known_main, known_short) == known_main);
  CHECK(decoderIgnoredTracksForGridSelection(
            true, true, seen, known_main, known_short) ==
        std::set<std::uint64_t>({1, 2, 3}));
}

void testShortRangeLookupAndPose() {
  const std::string path = fixturePath("short_range_grid.json");
  const MarkerGrid main = MarkerGrid::fromJson(path, 2);
  const ShortRangeMarkerGrid short_range =
      ShortRangeMarkerGrid::fromJson(path, main);
  std::vector<RelativeMarker> observations = {
      observation(3, 16, 6, 10), observation(0, 16, 5, 9),
      observation(2, 16, 6, 9), observation(1, 16, 5, 10)};
  const GridLookupResult lookup = short_range.lookup(observations);
  CHECK(lookup.status == GridLookupStatus::UNIQUE);
  CHECK(lookup.map_window_row == 0);
  CHECK(lookup.map_window_col == 0);
  CHECK(lookup.window_signature == std::vector<int>({16, 16, 16, 16}));
  CHECK(lookup.markers.size() == 4);
  CHECK(std::all_of(lookup.markers.begin(), lookup.markers.end(),
                    [](const GlobalMarker &marker) {
                      return marker.grid_type == "short_range" &&
                             marker.tile_i == 0 && marker.tile_j == 0 &&
                             marker.local_i >= 0 && marker.local_j >= 0;
                    }));

  const cv::Mat intrinsics = cameraMatrix();
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d world_to_camera = cv::Matx33d::eye();
  constexpr double distance = 1.0;
  const cv::Vec3d expected_camera_position(0.1, 0.1, -distance);
  const cv::Vec3d translation = -(world_to_camera * expected_camera_position);
  cv::Mat rvec;
  cv::Rodrigues(cv::Mat(world_to_camera), rvec);
  const cv::Mat tvec =
      (cv::Mat_<double>(3, 1) << translation[0], translation[1], translation[2]);

  std::vector<cv::Point3f> object_points;
  for (const ShortRangeMarker &marker : short_range.tiles()[0].markers) {
    object_points.push_back(marker.global_position);
  }
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, rvec, tvec, intrinsics, distortion,
                    image_points);
  std::vector<MarkerDetection> detections;
  for (std::size_t index : {std::size_t{2}, std::size_t{0}, std::size_t{3},
                            std::size_t{1}}) {
    detections.push_back({image_points[index].x, image_points[index].y,
                          short_range.tiles()[0].markers[index].id,
                          index + 1});
  }

  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;
  const LocalizationPipeline pipeline(path, 2, geometry);
  const LocalizationResult result = pipeline.localizeShortRange(
      detections, intrinsics, distortion, world_to_camera, distance);
  CHECK(result.status == LocalizationStatus::SUCCESS);
  CHECK(result.pose_valid);
  CHECK(result.grid_type == "short_range");
  CHECK(result.tile_i == 0);
  CHECK(result.tile_j == 0);
  CHECK(result.lookup.status == GridLookupStatus::UNIQUE);
  CHECK(result.pose_markers.size() == 4);
  CHECK(result.pnp_solver == "ippe_iterative");
  const cv::Vec3d recovered_camera_position(
      result.camera_position_world.at<double>(0, 0),
      result.camera_position_world.at<double>(1, 0),
      result.camera_position_world.at<double>(2, 0));
  CHECK_NEAR(cv::norm(recovered_camera_position - expected_camera_position),
             0.0, 2e-3);

  const LocalizationPipeline ap3p_pipeline(path, 2, geometry, true);
  const LocalizationResult ap3p_result = ap3p_pipeline.localizeShortRange(
      detections, intrinsics, distortion, world_to_camera, distance);
  CHECK(ap3p_result.status == LocalizationStatus::SUCCESS);
  CHECK(ap3p_result.pnp_solver == "ap3p");
  CHECK(ap3p_result.pose_markers.size() == 4);

  std::vector<MarkerDetection> mixed = detections;
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      const cv::Point3f point = main.cellToGlobal(row, col);
      std::vector<cv::Point2f> projected;
      cv::projectPoints(std::vector<cv::Point3f>{point}, rvec, tvec,
                        intrinsics, distortion, projected);
      mixed.push_back({projected[0].x, projected[0].y,
                       main.cells()[row][col]});
    }
  }
  std::vector<MarkerDetection> short_only;
  for (const MarkerDetection &detection : mixed) {
    if (short_range.markerIds().count(detection.id) != 0) {
      short_only.push_back(detection);
    }
  }
  CHECK(pipeline
            .localizeShortRange(short_only, intrinsics, distortion,
                                world_to_camera, distance)
            .lookup.status == GridLookupStatus::UNIQUE);
}

void testCrossGridIdAssignmentIsBidirectional() {
  const std::string path = fixturePath("short_range_grid.json");
  const MarkerGrid main = MarkerGrid::fromJson(path, 2);
  const ShortRangeMarkerGrid short_range =
      ShortRangeMarkerGrid::fromJson(path, main);
  const cv::Mat intrinsics = cameraMatrix(900.0, 880.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = rpyRotation({3.0, -4.0, 12.0});
  constexpr double distance = 0.42;
  const cv::Point2d camera_foot(-0.08, -0.09);
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;
  CrossGridIdAssigner assigner(main, short_range, geometry);

  std::vector<MarkerDetection> decoded_main;
  std::uint64_t track_id = 1;
  for (int row = 2; row <= 3; ++row) {
    for (int col = 2; col <= 3; ++col) {
      decoded_main.push_back(worldPointDetection(
          main.cellToGlobal(row, col), main.cells()[row][col], track_id++,
          camera_foot, distance, rotation, intrinsics, true));
    }
  }
  const CameraMapper main_mapper(main.cellSpacing(), geometry);
  const GridMappingResult main_mapping = main_mapper.detectionsToGrid(
      decoded_main, intrinsics, distortion, rotation, distance);
  CHECK(main_mapping.valid);
  const GridLookupResult main_lookup = main.lookup(main_mapping.markers);
  CHECK(main_lookup.status == GridLookupStatus::UNIQUE);
  CHECK(assigner.rememberUnique(main_lookup, decoded_main));

  const ShortRangeTile &tile = short_range.tiles()[1];
  std::vector<MarkerDetection> short_blobs;
  for (const ShortRangeMarker &marker : tile.markers) {
    short_blobs.push_back(worldPointDetection(
        marker.global_position, marker.id, track_id++, camera_foot, distance,
        rotation, intrinsics, false));
  }
  std::vector<MarkerDetection> overlap = decoded_main;
  overlap.insert(overlap.end(), short_blobs.begin(), short_blobs.end());
  const GridIdAssignmentResult inferred_short = assigner.assign(
      true, overlap, intrinsics, distortion, rotation, distance);
  CHECK(inferred_short.alignment_valid);
  CHECK(inferred_short.inferred_marker_count == 4);
  CHECK(inferred_short.detections.size() == 4);
  for (std::size_t index = 0; index < tile.markers.size(); ++index) {
    const MarkerDetection *detection =
        findTrack(inferred_short.detections, 5 + index);
    CHECK(detection != nullptr);
    CHECK(detection->id == tile.markers[index].id);
    CHECK(detection->inferred);
  }
  CHECK(findTrack(inferred_short.detections, 8)->id == 17);

  const CameraMapper short_mapper(short_range.cellSpacing(), geometry);
  const GridMappingResult short_mapping = short_mapper.detectionsToGrid(
      inferred_short.detections, intrinsics, distortion, rotation, distance);
  CHECK(short_mapping.valid);
  const GridLookupResult short_lookup =
      short_range.lookup(short_mapping.markers);
  CHECK(short_lookup.status == GridLookupStatus::UNIQUE);
  CHECK(assigner.rememberUnique(short_lookup, inferred_short.detections));
  assigner.forgetTracks({1, 2, 3, 4});

  const cv::Point2d moved_foot(-0.07, -0.075);
  std::vector<MarkerDetection> moved_overlap;
  for (std::size_t index = 0; index < tile.markers.size(); ++index) {
    moved_overlap.push_back(worldPointDetection(
        tile.markers[index].global_position, tile.markers[index].id,
        5 + index, moved_foot, distance, rotation, intrinsics, false));
  }
  std::uint64_t main_track = 20;
  for (int row = 2; row <= 3; ++row) {
    for (int col = 2; col <= 3; ++col) {
      moved_overlap.push_back(worldPointDetection(
          main.cellToGlobal(row, col), main.cells()[row][col], main_track++,
          moved_foot, distance, rotation, intrinsics, false));
    }
  }
  const GridIdAssignmentResult inferred_main = assigner.assign(
      false, moved_overlap, intrinsics, distortion, rotation, distance);
  CHECK(inferred_main.alignment_valid);
  CHECK(inferred_main.inferred_marker_count == 4);
  CHECK(inferred_main.detections.size() == 4);
  for (std::uint64_t id = 20; id < 24; ++id) {
    const MarkerDetection *detection = findTrack(inferred_main.detections, id);
    CHECK(detection != nullptr);
    const int cell = static_cast<int>(id - 20);
    CHECK(detection->map_row == 2 + cell / 2);
    CHECK(detection->map_col == 2 + cell % 2);
    CHECK(detection->id ==
          main.cells()[detection->map_row][detection->map_col]);
  }
}

void testCrossGridIdAssignmentRejectsIncompleteAndConflictingEvidence() {
  const std::string path = fixturePath("short_range_grid.json");
  const MarkerGrid main = MarkerGrid::fromJson(path, 2);
  const ShortRangeMarkerGrid short_range =
      ShortRangeMarkerGrid::fromJson(path, main);
  const cv::Mat intrinsics = cameraMatrix(900.0, 880.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = cv::Matx33d::eye();
  constexpr double distance = 0.4;
  const cv::Point2d camera_foot(0.1, 0.1);
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;

  std::vector<MarkerDetection> decoded_main;
  std::uint64_t track_id = 1;
  for (int row = 0; row <= 1; ++row) {
    for (int col = 0; col <= 1; ++col) {
      decoded_main.push_back(worldPointDetection(
          main.cellToGlobal(row, col), main.cells()[row][col], track_id++,
          camera_foot, distance, rotation, intrinsics, true));
    }
  }
  const CameraMapper main_mapper(main.cellSpacing(), geometry);
  const GridMappingResult mapping = main_mapper.detectionsToGrid(
      decoded_main, intrinsics, distortion, rotation, distance);
  CHECK(mapping.valid);
  const GridLookupResult lookup = main.lookup(mapping.markers);
  CHECK(lookup.status == GridLookupStatus::UNIQUE);

  CrossGridIdAssigner incomplete_assigner(main, short_range, geometry);
  CHECK(incomplete_assigner.rememberUnique(lookup, decoded_main));
  const ShortRangeTile &tile = short_range.tiles()[0];
  std::vector<MarkerDetection> conflicted = decoded_main;
  conflicted.push_back(worldPointDetection(
      tile.markers[0].global_position, tile.markers[0].id, 10, camera_foot,
      distance, rotation, intrinsics, false));
  conflicted.push_back(worldPointDetection(
      tile.markers[0].global_position, tile.markers[0].id, 11, camera_foot,
      distance, rotation, intrinsics, false));
  conflicted.push_back(worldPointDetection(
      tile.markers[1].global_position, tile.markers[1].id, 12, camera_foot,
      distance, rotation, intrinsics, false));
  conflicted.push_back(worldPointDetection(
      tile.markers[2].global_position, tile.markers[2].id, 13, camera_foot,
      distance, rotation, intrinsics, false));
  conflicted.push_back(worldPointDetection(
      tile.markers[3].global_position, tile.markers[3].id + 1, 14,
      camera_foot, distance, rotation, intrinsics, true));
  const GridIdAssignmentResult incomplete = incomplete_assigner.assign(
      true, conflicted, intrinsics, distortion, rotation, distance);
  CHECK(incomplete.alignment_valid);
  CHECK(incomplete.detections.empty());
  CHECK(incomplete.inferred_marker_count == 0);
  CHECK(!incomplete_assigner.hasGridTracks(true));

  CrossGridIdAssigner split_assigner(main, short_range, geometry);
  CHECK(split_assigner.rememberUnique(lookup, decoded_main));
  std::vector<MarkerDetection> split_anchors = {
      decoded_main[0],
      worldPointDetection(main.cellToGlobal(0, 1), main.cells()[0][1], 2,
                          {camera_foot.x + 0.02, camera_foot.y}, distance,
                          rotation, intrinsics, true)};
  for (std::size_t index = 0; index < tile.markers.size(); ++index) {
    split_anchors.push_back(worldPointDetection(
        tile.markers[index].global_position, tile.markers[index].id,
        30 + index, camera_foot, distance, rotation, intrinsics, false));
  }
  const GridIdAssignmentResult split = split_assigner.assign(
      true, split_anchors, intrinsics, distortion, rotation, distance);
  CHECK(!split.alignment_valid);
  CHECK(split.detections.empty());
  CHECK(split.message.find("disagree") != std::string::npos);

  GridLookupResult ambiguous = lookup;
  ambiguous.status = GridLookupStatus::AMBIGUOUS;
  CrossGridIdAssigner unseeded(main, short_range, geometry);
  CHECK(!unseeded.rememberUnique(ambiguous, decoded_main));
  CHECK(unseeded
            .assign(true, split_anchors, intrinsics, distortion, rotation,
                    distance)
            .detections.empty());
}

void testCenteredRectangularGridCoordinates() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("centered_rectangular_grid.json"), 2);
  CHECK(grid.rows() == 3);
  CHECK(grid.cols() == 5);
  CHECK_NEAR(grid.cellSpacing(), 0.2, 1e-7);

  const cv::Point3f top_left = grid.cellToGlobal(0, 0);
  const cv::Point3f one_col_right = grid.cellToGlobal(0, 1);
  const cv::Point3f one_row_down = grid.cellToGlobal(1, 0);
  const cv::Point3f center = grid.cellToGlobal(1, 2);
  const cv::Point3f bottom_right = grid.cellToGlobal(2, 4);

  CHECK_NEAR(grid.gridOrigin().x, top_left.x, 1e-7);
  CHECK_NEAR(grid.gridOrigin().y, top_left.y, 1e-7);
  CHECK_NEAR(grid.gridOrigin().z, top_left.z, 1e-7);
  CHECK_NEAR(top_left.x, 0.2, 1e-7);
  CHECK_NEAR(top_left.y, 0.4, 1e-7);
  CHECK_NEAR(one_col_right.x, top_left.x, 1e-7);
  CHECK_NEAR(one_col_right.y - top_left.y, -0.2, 1e-7);
  CHECK_NEAR(one_row_down.x - top_left.x, -0.2, 1e-7);
  CHECK_NEAR(one_row_down.y, top_left.y, 1e-7);
  CHECK_NEAR(top_left.z, 0.0, 1e-7);
  CHECK_NEAR(one_col_right.z, top_left.z, 1e-7);
  CHECK_NEAR(one_row_down.z, top_left.z, 1e-7);
  CHECK_NEAR(center.x, 0.0, 1e-7);
  CHECK_NEAR(center.y, 0.0, 1e-7);
  CHECK_NEAR(center.z, top_left.z, 1e-7);
  CHECK_NEAR(bottom_right.z, top_left.z, 1e-7);
  CHECK_NEAR((top_left.x + bottom_right.x) * 0.5, 0.0, 1e-7);
  CHECK_NEAR((top_left.y + bottom_right.y) * 0.5, 0.0, 1e-7);
  CHECK_NEAR((top_left.z + bottom_right.z) * 0.5, 0.0, 1e-7);
}

void testMarkerObservationFreshness() {
  CHECK(isMarkerObservationEligible(true, 0.0, 0.0));
  CHECK(!isMarkerObservationEligible(false, 0.01, 0.0));
  CHECK(isMarkerObservationEligible(false, 0.05, 0.05));
  CHECK(!isMarkerObservationEligible(false, 0.051, 0.05));
  CHECK(!isMarkerObservationEligible(true, -0.01, 0.05));
  CHECK(!isMarkerObservationEligible(
      true, std::numeric_limits<double>::quiet_NaN(), 0.05));
}

void testWindowTwoWithDuplicateIdsAndShuffledInput() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  auto markers = windowObservations(grid, 0, 0, 2, 4, 7);
  CHECK(markers[0].id == 0);
  CHECK(markers[1].id == 0);
  markers = reordered(markers, {2, 0, 3, 1});
  markers.push_back(observation(99, 17, 4, 7, false));

  const GridLookupResult result = grid.lookup(markers);
  CHECK(result.status == GridLookupStatus::UNIQUE);
  CHECK(result.required_marker_count == 4);
  CHECK(result.accepted_marker_count == 4);
  CHECK(result.candidate_count == 1);
  CHECK(result.best_match_count == 4);
  CHECK(result.relative_window_row == 4);
  CHECK(result.relative_window_col == 7);
  CHECK(result.map_window_row == 0);
  CHECK(result.map_window_col == 0);
  CHECK(result.window_signature == std::vector<int>({0, 0, 4, 5}));
  CHECK(result.markers.size() == 4);
  for (const GlobalMarker &marker : result.markers) {
    CHECK(marker.map_row == marker.relative_row - 4);
    CHECK(marker.map_col == marker.relative_col - 7);
    CHECK(grid.cells()[marker.map_row][marker.map_col] == marker.id);
    CHECK_NEAR(marker.global_x, 1.0 - marker.map_row * 0.25, 1e-7);
    CHECK_NEAR(marker.global_y, -2.0 - marker.map_col * 0.25, 1e-7);
  }

  for (int map_row = 0; map_row <= grid.rows() - grid.windowSize(); ++map_row) {
    for (int map_col = 0; map_col <= grid.cols() - grid.windowSize();
         ++map_col) {
      auto every_window = windowObservations(grid, map_row, map_col, 2,
                                             10 - map_row, -7 - map_col);
      std::reverse(every_window.begin(), every_window.end());
      const GridLookupResult every_result = grid.lookup(every_window);
      CHECK(every_result.status == GridLookupStatus::UNIQUE);
      CHECK(every_result.map_window_row == map_row);
      CHECK(every_result.map_window_col == map_col);
      CHECK(every_result.markers.size() == 4);
    }
  }
}

void testWindowThreeWithShuffledInput() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 3);
  CHECK(grid.totalWindowCount() == 9);
  CHECK(grid.uniqueWindowCount() == 9);
  auto markers = windowObservations(grid, 1, 1, 3, -2, 6);
  CHECK(std::count_if(
            markers.begin(), markers.end(),
            [](const RelativeMarker &marker) { return marker.id == 0; }) == 2);
  markers = reordered(markers, {8, 2, 4, 0, 7, 1, 5, 3, 6});

  const GridLookupResult result = grid.lookup(markers);
  CHECK(result.status == GridLookupStatus::UNIQUE);
  CHECK(result.required_marker_count == 9);
  CHECK(result.accepted_marker_count == 9);
  CHECK(result.best_match_count == 9);
  CHECK(result.relative_window_row == -2);
  CHECK(result.relative_window_col == 6);
  CHECK(result.map_window_row == 1);
  CHECK(result.map_window_col == 1);
  CHECK(result.markers.size() == 9);
  for (const GlobalMarker &marker : result.markers) {
    CHECK(marker.map_row == marker.relative_row + 3);
    CHECK(marker.map_col == marker.relative_col - 5);
  }
}

void testLookupFailuresAndAmbiguity() {
  const MarkerGrid unique =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);

  const std::vector<RelativeMarker> three_markers = {observation(0, 0, 0, 0),
                                                     observation(1, 0, 0, 1),
                                                     observation(2, 4, 1, 0)};
  const GridLookupResult insufficient = unique.lookup(three_markers);
  CHECK(insufficient.status == GridLookupStatus::INSUFFICIENT_MARKERS);
  CHECK(insufficient.required_marker_count == 4);

  const std::vector<RelativeMarker> sparse = {
      observation(0, 0, 0, 0), observation(1, 0, 0, 1), observation(2, 4, 2, 0),
      observation(3, 5, 2, 1)};
  CHECK(unique.lookup(sparse).status == GridLookupStatus::NO_COMPLETE_WINDOW);

  const std::vector<RelativeMarker> unknown = {
      observation(0, 99, 0, 0), observation(1, 99, 0, 1),
      observation(2, 99, 1, 0), observation(3, 99, 1, 1)};
  CHECK(unique.lookup(unknown).status == GridLookupStatus::NO_MATCH);

  const std::vector<RelativeMarker> duplicate_cell = {
      observation(0, 0, 0, 0), observation(1, 1, 0, 0), observation(2, 0, 0, 1),
      observation(3, 4, 1, 0), observation(4, 5, 1, 1)};
  const GridLookupResult duplicate_result = unique.lookup(duplicate_cell);
  CHECK(duplicate_result.status == GridLookupStatus::NO_MATCH);
  CHECK(duplicate_result.message.find("multiple detections") !=
        std::string::npos);

  auto contradictory = windowObservations(unique, 0, 0, 2, 0, 0);
  contradictory.push_back(observation(4, 1, 3, 3));
  const GridLookupResult contradictory_result = unique.lookup(contradictory);
  CHECK(contradictory_result.status == GridLookupStatus::NO_MATCH);
  CHECK(contradictory_result.best_match_count == 4);
  CHECK(contradictory_result.accepted_marker_count == 5);
  CHECK(contradictory_result.message.find("4/5") != std::string::npos);

  const MarkerGrid ambiguous =
      MarkerGrid::fromJson(fixturePath("ambiguous_grid.json"), 2);
  CHECK(ambiguous.maxWindowOccurrences() == 4);
  const GridLookupResult ambiguous_result =
      ambiguous.lookup(windowObservations(ambiguous, 0, 0, 2, 0, 0));
  CHECK(ambiguous_result.status == GridLookupStatus::AMBIGUOUS);
  CHECK(ambiguous_result.candidate_count == 4);
  CHECK(ambiguous_result.best_match_count == 4);
}

void testNormalizationWithNinetyDegreeOrientationAndNoise() {
  constexpr double spacing = 0.1;
  constexpr double distance = 2.0;
  const cv::Mat intrinsics = cameraMatrix();
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;
  const cv::Matx33d rotation = rpyRotation({0.0, 0.0, 90.0});
  const CameraMapper mapper(spacing, geometry);

  struct Sample {
    int row;
    int col;
    int id;
    cv::Point2f noise;
  };
  const std::array<Sample, 4> samples = {{{1, 1, 13, {0.4F, -0.3F}},
                                          {0, 0, 10, {-0.5F, 0.2F}},
                                          {1, 0, 12, {0.2F, 0.5F}},
                                          {0, 1, 11, {-0.3F, -0.4F}}}};
  std::vector<MarkerDetection> detections;
  for (const Sample &sample : samples) {
    cv::Point2f pixel = projectPlanePoint(-(4 + sample.row) * spacing,
                                          -(2 + sample.col) * spacing, distance,
                                          rotation, intrinsics);
    pixel += sample.noise;
    detections.push_back({pixel.x, pixel.y, sample.id});
  }

  const cv::Point2f base =
      projectPlanePoint(-0.4, -0.2, distance, rotation, intrinsics);
  const cv::Point2f one_col =
      projectPlanePoint(-0.4, -0.3, distance, rotation, intrinsics);
  const cv::Point2f one_row =
      projectPlanePoint(-0.5, -0.2, distance, rotation, intrinsics);
  CHECK(one_col.x > base.x);
  CHECK_NEAR(one_col.y, base.y, 1e-4);
  CHECK_NEAR(one_row.x, base.x, 1e-4);
  CHECK(one_row.y < base.y);

  const GridMappingResult result =
      mapper.detectionsToGrid(detections, intrinsics, distortion, rotation,
                              distance);
  CHECK(result.valid);
  CHECK(result.markers.size() == samples.size());
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const RelativeMarker &marker = result.markers[index];
    CHECK(marker.detection_index == index);
    CHECK(marker.id == samples[index].id);
    CHECK(marker.row == samples[index].row);
    CHECK(marker.col == samples[index].col);
    CHECK(marker.accepted);
    CHECK(marker.row_rounding_error < 0.04F);
    CHECK(marker.col_rounding_error < 0.04F);
  }
}

void testQuantizationRejectsOffGridDetection() {
  constexpr double spacing = 0.1;
  constexpr double distance = 2.0;
  const cv::Mat intrinsics = cameraMatrix();
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.20;
  const CameraMapper mapper(spacing, geometry);
  const cv::Matx33d rotation = cv::Matx33d::eye();

  struct PlaneSample {
    double x;
    double y;
    int id;
  };
  const std::array<PlaneSample, 5> samples = {{{0.0, 0.0, 1},
                                               {0.1, 0.0, 2},
                                               {0.0, 0.1, 3},
                                               {0.1, 0.1, 4},
                                               {0.2, 0.04, 5}}};
  std::vector<MarkerDetection> detections;
  for (const PlaneSample &sample : samples) {
    const cv::Point2f pixel =
        projectPlanePoint(sample.x, sample.y, distance, rotation, intrinsics);
    detections.push_back({pixel.x, pixel.y, sample.id});
  }

  const GridMappingResult result =
      mapper.detectionsToGrid(detections, intrinsics, distortion, rotation,
                              distance);
  CHECK(result.valid);
  CHECK(result.markers.size() == 5);
  CHECK(std::count_if(
            result.markers.begin(), result.markers.end(),
            [](const RelativeMarker &marker) { return marker.accepted; }) == 4);
  for (std::size_t index = 0; index < 4; ++index) {
    CHECK(result.markers[index].accepted);
  }
  CHECK(!result.markers.back().accepted);
  CHECK(result.markers.back().col_rounding_error > geometry.rounding_tolerance);
}

void testGridIdAssignmentBootstrapsAndLocalizesSameFrame() {
  const std::string map_path = fixturePath("unique_grid.json");
  const MarkerGrid grid = MarkerGrid::fromJson(map_path, 2);
  const cv::Mat intrinsics = cameraMatrix(900.0, 880.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = rpyRotation({4.0, -6.0, 18.0});
  constexpr double distance = 2.2;
  constexpr double row_offset = 0.23;
  constexpr double col_offset = -0.31;
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.16;
  GridIdAssigner assigner(grid, geometry);

  std::vector<MarkerDetection> decoded;
  std::vector<MarkerDetection> current;
  std::uint64_t track_id = 1;
  for (int row = 1; row <= 2; ++row) {
    for (int col = 1; col <= 2; ++col) {
      const MarkerDetection detection = mapCellDetection(
          grid, row, col, track_id++, row_offset, col_offset, distance,
          rotation, intrinsics, true);
      decoded.push_back(detection);
      current.push_back(detection);
    }
  }
  current.push_back(mapCellDetection(grid, 1, 3, track_id++, row_offset,
                                     col_offset, distance, rotation, intrinsics,
                                     false));
  current.push_back(mapCellDetection(grid, 2, 3, track_id++, row_offset,
                                     col_offset, distance, rotation, intrinsics,
                                     false));
  current.push_back(gridCoordinateDetection(
      3.37 + row_offset, 2.34 + col_offset, -1, track_id++,
      grid.cellSpacing(), distance, rotation, intrinsics));
  current.push_back(mapCellDetection(grid, 1, 4, track_id++, row_offset,
                                     col_offset, distance, rotation, intrinsics,
                                     false));

  const GridIdAssignmentResult assigned = assigner.assign(
      decoded, current, intrinsics, distortion, rotation, distance);
  CHECK(assigned.map_locked);
  CHECK(assigned.alignment_valid);
  CHECK(assigned.detections.size() == 6);
  CHECK(assigned.inferred_marker_count == 2);
  CHECK(assigned.rejected_blob_count == 2);
  const MarkerDetection *right_top = findTrack(assigned.detections, 5);
  const MarkerDetection *right_bottom = findTrack(assigned.detections, 6);
  CHECK(right_top != nullptr);
  CHECK(right_bottom != nullptr);
  CHECK(right_top->inferred);
  CHECK(right_top->map_row == 1);
  CHECK(right_top->map_col == 3);
  CHECK(right_top->id == grid.cells()[1][3]);
  CHECK(right_bottom->map_row == 2);
  CHECK(right_bottom->map_col == 3);
  CHECK(right_bottom->id == grid.cells()[2][3]);

  const LocalizationPipeline pipeline(map_path, 2, geometry);
  const LocalizationResult localization = pipeline.localize(
      assigned.detections, intrinsics, distortion, rotation, distance);
  CHECK(localization.status == LocalizationStatus::SUCCESS);
  CHECK(localization.pose_valid);
  CHECK(localization.lookup.markers.size() == 6);
  CHECK(localization.lookup.complete_window_count == 0);

  const std::vector<MarkerDetection> next_frontier = {
      mapCellDetection(grid, 1, 3, 5, 0.31, -0.18, distance, rotation,
                       intrinsics, false),
      mapCellDetection(grid, 1, 4, 8, 0.31, -0.18, distance, rotation,
                       intrinsics, false)};
  const GridIdAssignmentResult expanded = assigner.assign(
      {}, next_frontier, intrinsics, distortion, rotation, distance);
  CHECK(expanded.detections.size() == 2);
  CHECK(findTrack(expanded.detections, 8) != nullptr);
}

void testGridIdAssignmentNeedsACompleteDecodedWindow() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  const cv::Mat intrinsics = cameraMatrix();
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = cv::Matx33d::eye();
  constexpr double distance = 2.0;
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;
  GridIdAssigner assigner(grid, geometry);

  std::vector<MarkerDetection> current;
  for (int index = 0; index < 4; ++index) {
    const int row = 1 + index / 2;
    const int col = 1 + index % 2;
    current.push_back(mapCellDetection(
        grid, row, col, index + 1, 0.2, -0.1, distance, rotation, intrinsics,
        index < 3));
  }
  const std::vector<MarkerDetection> decoded(current.begin(),
                                              current.begin() + 3);
  const GridIdAssignmentResult assigned = assigner.assign(
      decoded, current, intrinsics, distortion, rotation, distance);
  CHECK(!assigned.map_locked);
  CHECK(!assigned.alignment_valid);
  CHECK(assigned.detections.size() == 3);
  CHECK(findTrack(assigned.detections, 4) == nullptr);

  current.back().id = grid.cells()[2][2];
  CHECK(assigner
            .assign(current, current, intrinsics, distortion, rotation,
                    distance)
            .map_locked);
  assigner.forgetTracks({1, 2, 3, 4});
  CHECK(!assigner.mapLocked());
}

void testGridIdAssignmentPropagatesAcrossCameraMotion() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  const cv::Mat intrinsics = cameraMatrix(820.0, 810.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = rpyRotation({2.0, -3.0, 11.0});
  constexpr double distance = 2.4;
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.16;
  GridIdAssigner assigner(grid, geometry);

  std::vector<MarkerDetection> decoded;
  std::vector<MarkerDetection> first_frame;
  std::uint64_t track_id = 1;
  for (int row = 1; row <= 2; ++row) {
    for (int col = 1; col <= 2; ++col) {
      const MarkerDetection detection = mapCellDetection(
          grid, row, col, track_id++, 0.15, -0.22, distance, rotation,
          intrinsics, true);
      decoded.push_back(detection);
      first_frame.push_back(detection);
    }
  }
  first_frame.push_back(mapCellDetection(grid, 1, 3, 5, 0.15, -0.22,
                                         distance, rotation, intrinsics,
                                         false));
  first_frame.push_back(mapCellDetection(grid, 2, 3, 6, 0.15, -0.22,
                                         distance, rotation, intrinsics,
                                         false));
  CHECK(assigner
            .assign(decoded, first_frame, intrinsics, distortion, rotation,
                    distance)
            .detections.size() == 6);

  const cv::Matx33d moved_rotation = rpyRotation({5.0, -7.0, 22.0});
  constexpr double moved_distance = 2.25;
  std::vector<MarkerDetection> moved_frame = {
      mapCellDetection(grid, 1, 3, 5, 0.33, -0.08, moved_distance,
                       moved_rotation,
                       intrinsics, false),
      mapCellDetection(grid, 2, 3, 6, 0.33, -0.08, moved_distance,
                       moved_rotation,
                       intrinsics, false),
      mapCellDetection(grid, 1, 4, 7, 0.33, -0.08, moved_distance,
                       moved_rotation,
                       intrinsics, false),
      mapCellDetection(grid, 2, 4, 8, 0.33, -0.08, moved_distance,
                       moved_rotation,
                       intrinsics, false)};
  for (int index = 0; index < 12; ++index) {
    moved_frame.push_back(gridCoordinateDetection(
        0.35 + (index % 4) + 0.33, 0.38 + (index / 4) - 0.08, -1,
        100 + index, grid.cellSpacing(), moved_distance, moved_rotation,
        intrinsics));
  }

  const GridIdAssignmentResult moved = assigner.assign(
      {}, moved_frame, intrinsics, distortion, moved_rotation, distance);
  CHECK(moved.alignment_valid);
  CHECK(moved.detections.size() == 4);
  CHECK(moved.rejected_blob_count == 12);
  const MarkerDetection *new_top = findTrack(moved.detections, 7);
  const MarkerDetection *new_bottom = findTrack(moved.detections, 8);
  CHECK(new_top != nullptr);
  CHECK(new_bottom != nullptr);
  CHECK(new_top->map_row == 1);
  CHECK(new_top->map_col == 4);
  CHECK(new_top->id == grid.cells()[1][4]);
  CHECK(new_bottom->map_row == 2);
  CHECK(new_bottom->map_col == 4);
  CHECK(new_bottom->id == grid.cells()[2][4]);

  const std::vector<MarkerDetection> one_anchor_frame = {
      mapCellDetection(grid, 1, 4, 7, 0.41, 0.07, moved_distance,
                       moved_rotation,
                       intrinsics, false),
      mapCellDetection(grid, 0, 4, 9, 0.41, 0.07, moved_distance,
                       moved_rotation,
                       intrinsics, false)};
  const GridIdAssignmentResult one_anchor = assigner.assign(
      {}, one_anchor_frame, intrinsics, distortion, moved_rotation,
      moved_distance);
  CHECK(one_anchor.alignment_valid);
  CHECK(one_anchor.detections.size() == 2);
  const MarkerDetection *new_from_one_anchor =
      findTrack(one_anchor.detections, 9);
  CHECK(new_from_one_anchor != nullptr);
  CHECK(new_from_one_anchor->map_row == 0);
  CHECK(new_from_one_anchor->map_col == 4);
  CHECK(new_from_one_anchor->id == grid.cells()[0][4]);
}

void testGridIdAssignmentRejectsConflictsAndDuplicateCells() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  const cv::Mat intrinsics = cameraMatrix();
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = cv::Matx33d::eye();
  constexpr double distance = 2.0;
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;
  GridIdAssigner assigner(grid, geometry);

  std::vector<MarkerDetection> decoded;
  std::vector<MarkerDetection> first_frame;
  std::uint64_t track_id = 1;
  for (int row = 1; row <= 2; ++row) {
    for (int col = 1; col <= 2; ++col) {
      const MarkerDetection detection = mapCellDetection(
          grid, row, col, track_id++, 0.1, -0.2, distance, rotation,
          intrinsics, true);
      decoded.push_back(detection);
      first_frame.push_back(detection);
    }
  }
  first_frame.push_back(mapCellDetection(grid, 1, 3, 5, 0.1, -0.2,
                                         distance, rotation, intrinsics,
                                         false));
  CHECK(assigner
            .assign(decoded, first_frame, intrinsics, distortion, rotation,
                    distance)
            .detections.size() == 5);

  std::vector<MarkerDetection> current;
  for (int index = 0; index < 3; ++index) {
    const int row = 1 + index / 2;
    const int col = 1 + index % 2;
    current.push_back(mapCellDetection(grid, row, col, index + 1, 0.24, -0.05,
                                       distance, rotation, intrinsics, true));
  }
  current.push_back(gridCoordinateDetection(
      1.24 + 0.34, 3.0 - 0.05, -1, 5, grid.cellSpacing(), distance,
      rotation, intrinsics));
  current.push_back(mapCellDetection(grid, 2, 3, 6, 0.24, -0.05, distance,
                                     rotation, intrinsics, false));
  current.push_back(mapCellDetection(grid, 2, 3, 7, 0.24, -0.05, distance,
                                     rotation, intrinsics, false));
  MarkerDetection conflict = mapCellDetection(
      grid, 3, 2, 8, 0.24, -0.05, distance, rotation, intrinsics, true);
  conflict.id = (conflict.id + 1) % grid.numIds();
  current.push_back(conflict);

  const GridIdAssignmentResult result = assigner.assign(
      {conflict}, current, intrinsics, distortion, rotation, distance);
  CHECK(result.alignment_valid);
  CHECK(result.detections.size() == 3);
  CHECK(result.rejected_blob_count == 4);
  CHECK(findTrack(result.detections, 5) == nullptr);
  CHECK(findTrack(result.detections, 6) == nullptr);
  CHECK(findTrack(result.detections, 7) == nullptr);
  CHECK(findTrack(result.detections, 8) == nullptr);

  std::vector<MarkerDetection> decoder_wins;
  for (int index = 0; index < 3; ++index) {
    const int row = 1 + index / 2;
    const int col = 1 + index % 2;
    decoder_wins.push_back(mapCellDetection(
        grid, row, col, index + 1, 0.3, 0.04, distance, rotation, intrinsics,
        true));
  }
  decoder_wins.push_back(mapCellDetection(grid, 1, 3, 5, 0.3, 0.04,
                                          distance, rotation, intrinsics,
                                          false));
  const MarkerDetection confirmed = mapCellDetection(
      grid, 1, 3, 9, 0.3, 0.04, distance, rotation, intrinsics, true);
  decoder_wins.push_back(confirmed);
  const GridIdAssignmentResult replacement = assigner.assign(
      {confirmed}, decoder_wins, intrinsics, distortion, rotation, distance);
  CHECK(findTrack(replacement.detections, 5) == nullptr);
  CHECK(findTrack(replacement.detections, 9) != nullptr);

  std::vector<MarkerDetection> decoded_collision(decoder_wins.begin(),
                                                  decoder_wins.begin() + 3);
  decoded_collision.push_back(confirmed);
  const MarkerDetection second_confirmed = mapCellDetection(
      grid, 1, 3, 10, 0.3, 0.04, distance, rotation, intrinsics, true);
  decoded_collision.push_back(second_confirmed);
  const GridIdAssignmentResult ambiguous = assigner.assign(
      {second_confirmed}, decoded_collision, intrinsics, distortion, rotation,
      distance);
  CHECK(findTrack(ambiguous.detections, 9) == nullptr);
  CHECK(findTrack(ambiguous.detections, 10) == nullptr);
}

void testGridIdAssignmentRejectsSplitAnchorConsensus() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  const cv::Mat intrinsics = cameraMatrix();
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = cv::Matx33d::eye();
  constexpr double distance = 2.0;
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;
  GridIdAssigner assigner(grid, geometry);

  std::vector<MarkerDetection> decoded;
  for (int row = 1; row <= 2; ++row) {
    for (int col = 1; col <= 2; ++col) {
      decoded.push_back(mapCellDetection(
          grid, row, col, decoded.size() + 1, 0.1, -0.2, distance, rotation,
          intrinsics, true));
    }
  }
  CHECK(assigner
            .assign(decoded, decoded, intrinsics, distortion, rotation,
                    distance)
            .map_locked);

  std::vector<MarkerDetection> split;
  split.push_back(mapCellDetection(grid, 1, 1, 1, 0.1, -0.2, distance,
                                   rotation, intrinsics, false));
  split.push_back(mapCellDetection(grid, 1, 2, 2, 0.1, -0.2, distance,
                                   rotation, intrinsics, false));
  split.push_back(mapCellDetection(grid, 2, 1, 3, 0.8, 0.5, distance,
                                   rotation, intrinsics, false));
  split.push_back(mapCellDetection(grid, 2, 2, 4, 0.8, 0.5, distance,
                                   rotation, intrinsics, false));
  const GridIdAssignmentResult result = assigner.assign(
      {}, split, intrinsics, distortion, rotation, distance);
  CHECK(!result.alignment_valid);
  CHECK(result.detections.empty());
  CHECK(result.message.find("disagree") != std::string::npos);

  GridIdAssigner spread_assigner(grid, geometry);
  CHECK(spread_assigner
            .assign(decoded, decoded, intrinsics, distortion, rotation,
                    distance)
            .map_locked);
  const std::vector<MarkerDetection> spread = {
      mapCellDetection(grid, 1, 1, 1, 0.10, -0.2, distance, rotation,
                       intrinsics, false),
      mapCellDetection(grid, 1, 2, 2, 0.38, -0.2, distance, rotation,
                       intrinsics, false),
      mapCellDetection(grid, 2, 1, 3, 0.66, -0.2, distance, rotation,
                       intrinsics, false)};
  const GridIdAssignmentResult spread_result = spread_assigner.assign(
      {}, spread, intrinsics, distortion, rotation, distance);
  CHECK(!spread_result.alignment_valid);
  CHECK(spread_result.detections.empty());
}

void testDecodedWindowRecoversFromAStaleInferredAnchor() {
  const MarkerGrid grid =
      MarkerGrid::fromJson(fixturePath("unique_grid.json"), 2);
  const cv::Mat intrinsics = cameraMatrix();
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = cv::Matx33d::eye();
  constexpr double distance = 2.0;
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;
  GridIdAssigner assigner(grid, geometry);

  std::vector<MarkerDetection> first_window;
  for (int row = 0; row <= 1; ++row) {
    for (int col = 0; col <= 1; ++col) {
      first_window.push_back(mapCellDetection(
          grid, row, col, first_window.size() + 1, 0.1, -0.1, distance,
          rotation, intrinsics, true));
    }
  }
  CHECK(assigner
            .assign(first_window, first_window, intrinsics, distortion,
                    rotation, distance)
            .map_locked);

  std::vector<MarkerDetection> incompatible_window;
  for (int row = 3; row <= 4; ++row) {
    for (int col = 3; col <= 4; ++col) {
      incompatible_window.push_back(mapCellDetection(
          grid, row, col, 10 + incompatible_window.size(), 0.8, 0.6,
          distance, rotation, intrinsics, true));
    }
  }
  std::vector<MarkerDetection> current = {
      mapCellDetection(grid, 0, 0, 1, 0.1, -0.1, distance, rotation,
                       intrinsics, false)};
  current.insert(current.end(), incompatible_window.begin(),
                 incompatible_window.end());
  const GridIdAssignmentResult result = assigner.assign(
      incompatible_window, current, intrinsics, distortion, rotation,
      distance);
  CHECK(result.alignment_valid);
  CHECK(result.detections.size() == 4);
  CHECK(findTrack(result.detections, 1) == nullptr);
  for (std::uint64_t track_id = 10; track_id < 14; ++track_id) {
    CHECK(findTrack(result.detections, track_id) != nullptr);
  }

  std::vector<MarkerDetection> overlapping_relock;
  std::uint64_t replacement_track = 10;
  for (int row = 0; row <= 1; ++row) {
    for (int col = 2; col <= 3; ++col) {
      overlapping_relock.push_back(mapCellDetection(
          grid, row, col, replacement_track, 0.35, -0.25, distance, rotation,
          intrinsics, true));
      replacement_track =
          replacement_track == 10 ? 20 : replacement_track + 1;
    }
  }
  const GridIdAssignmentResult overlapping = assigner.assign(
      overlapping_relock, overlapping_relock, intrinsics, distortion,
      rotation, distance);
  CHECK(overlapping.alignment_valid);
  CHECK(overlapping.detections.size() == 4);
  const MarkerDetection *remapped = findTrack(overlapping.detections, 10);
  CHECK(remapped != nullptr);
  CHECK(remapped->map_row == 0);
  CHECK(remapped->map_col == 2);

  std::vector<MarkerDetection> mixed_overlap_relock = {
      mapCellDetection(grid, 1, 2, 21, 0.18, 0.12, distance, rotation,
                       intrinsics, true),
      mapCellDetection(grid, 1, 3, 10, 0.18, 0.12, distance, rotation,
                       intrinsics, true),
      mapCellDetection(grid, 2, 2, 30, 0.18, 0.12, distance, rotation,
                       intrinsics, true),
      mapCellDetection(grid, 2, 3, 31, 0.18, 0.12, distance, rotation,
                       intrinsics, true)};
  const GridIdAssignmentResult mixed = assigner.assign(
      mixed_overlap_relock, mixed_overlap_relock, intrinsics, distortion,
      rotation, distance);
  CHECK(mixed.alignment_valid);
  CHECK(mixed.detections.size() == 4);
  const MarkerDetection *mixed_consistent = findTrack(mixed.detections, 21);
  const MarkerDetection *mixed_remapped = findTrack(mixed.detections, 10);
  CHECK(mixed_consistent != nullptr);
  CHECK(mixed_remapped != nullptr);
  CHECK(mixed_consistent->map_row == 1);
  CHECK(mixed_consistent->map_col == 2);
  CHECK(mixed_remapped->map_row == 1);
  CHECK(mixed_remapped->map_col == 3);
}

void testMapAlignedLocalizationDoesNotNeedAnotherWindow() {
  const std::string map_path = fixturePath("unique_grid.json");
  const MarkerGrid grid = MarkerGrid::fromJson(map_path, 2);
  const cv::Mat intrinsics = cameraMatrix(900.0, 900.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
  const cv::Matx33d rotation = rpyRotation({1.0, -2.0, 8.0});
  constexpr double distance = 2.3;
  CameraPlaneGeometry geometry;
  geometry.rounding_tolerance = 0.15;

  std::vector<MarkerDetection> detections;
  const std::array<std::pair<int, int>, 4> sparse_cells = {
      std::pair<int, int>{0, 0}, {0, 2}, {2, 0}, {2, 2}};
  for (std::size_t index = 0; index < sparse_cells.size(); ++index) {
    const auto [row, col] = sparse_cells[index];
    MarkerDetection detection = mapCellDetection(
        grid, row, col, index + 1, 0.18, -0.27, distance, rotation,
        intrinsics, false);
    detection.id = grid.cells()[row][col];
    detection.inferred = true;
    detection.map_row = row;
    detection.map_col = col;
    detections.push_back(detection);
  }

  const LocalizationPipeline pipeline(map_path, 2, geometry);
  const LocalizationResult result = pipeline.localize(
      detections, intrinsics, distortion, rotation, distance);
  CHECK(result.status == LocalizationStatus::SUCCESS);
  CHECK(result.pose_valid);
  CHECK(result.lookup.status == GridLookupStatus::UNIQUE);
  CHECK(result.lookup.complete_window_count == 0);
  CHECK(result.lookup.markers.size() == 4);
}

void testSyntheticFullLocalizationRecoversPose() {
  constexpr int window_size = 3;
  const std::string map_path = fixturePath("unique_grid.json");
  const MarkerGrid grid = MarkerGrid::fromJson(map_path, window_size);
  const cv::Mat intrinsics = cameraMatrix(920.0, 900.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);

  CameraPlaneGeometry geometry;
  constexpr double distance = 2.3;
  geometry.rounding_tolerance = 0.15;
  const cv::Vec4d drone_quaternion =
      quaternionFromRpyDegrees({8.0, -12.0, 17.0});
  const auto dynamic_world_to_camera =
      pose_math::gridToCameraRotationFromDroneQuaternionXyzw(
          drone_quaternion);
  CHECK(dynamic_world_to_camera.has_value());
  const cv::Matx33d world_to_camera = *dynamic_world_to_camera;

  const cv::Point3f plane_foot = grid.cellToGlobal(2, 2);
  const cv::Vec3d expected_camera_position(plane_foot.x, plane_foot.y,
                                           plane_foot.z + distance);
  const cv::Vec3d translation = -(world_to_camera * expected_camera_position);
  cv::Mat rotation_vector;
  cv::Rodrigues(cv::Mat(world_to_camera), rotation_vector);
  const cv::Mat translation_vector = (cv::Mat_<double>(3, 1) << translation[0],
                                      translation[1], translation[2]);

  std::vector<cv::Point3f> object_points;
  std::vector<int> ids;
  for (int row = 1; row <= 3; ++row) {
    for (int col = 1; col <= 3; ++col) {
      object_points.push_back(grid.cellToGlobal(row, col));
      ids.push_back(grid.cells()[row][col]);
    }
  }
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, rotation_vector, translation_vector,
                    intrinsics, distortion, image_points);
  std::vector<MarkerDetection> detections;
  for (std::size_t reverse_index = image_points.size(); reverse_index > 0;
       --reverse_index) {
    const std::size_t index = reverse_index - 1;
    detections.push_back(
        {image_points[index].x, image_points[index].y, ids[index]});
  }

  const LocalizationPipeline pipeline(map_path, window_size, geometry);
  const LocalizationResult result =
      pipeline.localize(detections, intrinsics, distortion, world_to_camera,
                        distance);
  CHECK(result.status == LocalizationStatus::SUCCESS);
  CHECK(result.pose_valid);
  CHECK(result.lookup.status == GridLookupStatus::UNIQUE);
  CHECK(result.lookup.map_window_row == 1);
  CHECK(result.lookup.map_window_col == 1);
  CHECK(result.lookup.markers.size() == 9);
  CHECK(result.pose_markers.size() == 9);
  CHECK(result.pnp_solver == "ippe_iterative");
  CHECK_NEAR(result.reprojection_error, 0.0, 1e-2);
  CHECK_NEAR(result.distance_used, distance, 1e-12);
  CHECK_NEAR(result.camera_to_plane_distance, distance, 2e-3);

  const cv::Vec3d recovered_marker_position_camera(
      result.tvec_world_to_camera.at<double>(0, 0),
      result.tvec_world_to_camera.at<double>(1, 0),
      result.tvec_world_to_camera.at<double>(2, 0));
  CHECK_NEAR(cv::norm(recovered_marker_position_camera - translation), 0.0,
             2e-3);
  CHECK(std::abs(std::abs(recovered_marker_position_camera[2]) -
                 result.camera_to_plane_distance) > 0.5);

  const cv::Vec3d recovered_position(
      result.camera_position_world.at<double>(0, 0),
      result.camera_position_world.at<double>(1, 0),
      result.camera_position_world.at<double>(2, 0));
  CHECK_NEAR(cv::norm(recovered_position - expected_camera_position), 0.0,
             2e-3);
  const cv::Mat expected_camera_rotation = cv::Mat(world_to_camera.t());
  CHECK_NEAR(cv::norm(result.camera_rotation_world - expected_camera_rotation,
                      cv::NORM_L2),
             0.0, 1e-3);
}

void testCenterTwoByTwoAp3pUsesClosestWindow() {
  constexpr int window_size = 2;
  const std::string map_path = fixturePath("unique_grid.json");
  const MarkerGrid grid = MarkerGrid::fromJson(map_path, window_size);
  const cv::Size frame_size(1280, 720);
  const cv::Mat intrinsics = cameraMatrix(920.0, 900.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);

  CameraPlaneGeometry geometry;
  constexpr double distance = 2.3;
  geometry.rounding_tolerance = 0.15;

  const cv::Point3f selected_top_left = grid.cellToGlobal(2, 2);
  const cv::Matx33d world_to_camera = pose_math::cameraToDroneRotation().t();
  const cv::Vec3d expected_camera_position(
      selected_top_left.x - 0.5 * grid.cellSpacing(),
      selected_top_left.y - 0.5 * grid.cellSpacing(),
      selected_top_left.z + distance);
  cv::Mat rotation_vector;
  cv::Rodrigues(cv::Mat(world_to_camera), rotation_vector);
  const cv::Vec3d translation = -(world_to_camera * expected_camera_position);
  const cv::Mat translation_vector = (cv::Mat_<double>(3, 1)
      << translation[0], translation[1], translation[2]);

  std::vector<cv::Point3f> object_points;
  std::vector<int> ids;
  for (int row = 1; row <= 3; ++row) {
    for (int col = 1; col <= 3; ++col) {
      object_points.push_back(grid.cellToGlobal(row, col));
      ids.push_back(grid.cells()[row][col]);
    }
  }
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, rotation_vector, translation_vector,
                    intrinsics, distortion, image_points);

  std::vector<MarkerDetection> detections;
  const std::array<std::size_t, 9> order = {7, 0, 5, 2, 8, 3, 1, 6, 4};
  for (std::size_t index : order) {
    detections.push_back(
        {image_points[index].x, image_points[index].y, ids[index]});
  }

  const LocalizationPipeline pipeline(map_path, window_size, geometry, true);
  const auto dynamic_world_to_camera =
      pose_math::gridToCameraRotationFromDroneQuaternionXyzw(
          {0.0, 0.0, 0.0, 1.0});
  CHECK(dynamic_world_to_camera.has_value());
  const LocalizationResult result =
      pipeline.localize(detections, intrinsics, distortion,
                        *dynamic_world_to_camera, distance, frame_size);
  CHECK(result.status == LocalizationStatus::SUCCESS);
  CHECK(result.pose_valid);
  CHECK(result.pnp_solver == "ap3p");
  CHECK(result.lookup.markers.size() == 9);
  CHECK(result.pose_markers.size() == 4);
  CHECK(result.pose_markers[0].map_row == 2);
  CHECK(result.pose_markers[0].map_col == 2);
  CHECK(result.pose_markers[1].map_row == 2);
  CHECK(result.pose_markers[1].map_col == 3);
  CHECK(result.pose_markers[2].map_row == 3);
  CHECK(result.pose_markers[2].map_col == 2);
  CHECK(result.pose_markers[3].map_row == 3);
  CHECK(result.pose_markers[3].map_col == 3);
  CHECK(result.reprojection_error < 1e-2);

  const LocalizationResult missing_size =
      pipeline.localize(detections, intrinsics, distortion,
                        *dynamic_world_to_camera, distance);
  CHECK(missing_size.status == LocalizationStatus::PNP_FAILED);
  CHECK(missing_size.message.find("image dimensions") != std::string::npos);
  CHECK_THROWS_CONTAINING(
      LocalizationPipeline(map_path, 3, geometry, true), "window_size 2");
}

void testNearFrontoParallelIppeUsesKnownGeometry() {
  constexpr int window_size = 2;
  const std::string map_path = fixturePath("unique_grid.json");
  const MarkerGrid grid = MarkerGrid::fromJson(map_path, window_size);
  const cv::Mat intrinsics = cameraMatrix(1180.0, 1160.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);

  CameraPlaneGeometry geometry;
  constexpr double distance = 2.5;
  geometry.rounding_tolerance = 0.15;
  const cv::Matx33d world_to_camera = rpyRotation({0.35, -0.25, 6.0});

  const int map_row = 1;
  const int map_col = 2;
  const cv::Point3f top_left = grid.cellToGlobal(map_row, map_col);
  const cv::Vec3d expected_camera_position(
      top_left.x + 0.5 * grid.cellSpacing(),
      top_left.y + 0.5 * grid.cellSpacing(), top_left.z - distance);
  const cv::Vec3d translation = -(world_to_camera * expected_camera_position);
  cv::Mat expected_rvec;
  cv::Rodrigues(cv::Mat(world_to_camera), expected_rvec);
  const cv::Mat expected_tvec = (cv::Mat_<double>(3, 1) << translation[0],
                                 translation[1], translation[2]);

  std::vector<cv::Point3f> object_points;
  std::vector<int> ids;
  for (int dr = 0; dr < window_size; ++dr) {
    for (int dc = 0; dc < window_size; ++dc) {
      object_points.push_back(grid.cellToGlobal(map_row + dr, map_col + dc));
      ids.push_back(grid.cells()[map_row + dr][map_col + dc]);
    }
  }
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, expected_rvec, expected_tvec, intrinsics,
                    distortion, image_points);
  const std::array<cv::Point2f, 4> noise = {
      cv::Point2f{-0.10F, 0.06F}, cv::Point2f{0.09F, -0.07F},
      cv::Point2f{0.05F, 0.11F}, cv::Point2f{-0.08F, -0.09F}};
  for (std::size_t i = 0; i < image_points.size(); ++i) {
    image_points[i] += noise[i];
  }

  // IPPE must expose both branches for this nearly fronto-parallel planar
  // view; the pipeline should choose the one consistent with known geometry.
  std::vector<cv::Mat> ippe_rvecs;
  std::vector<cv::Mat> ippe_tvecs;
  CHECK(cv::solvePnPGeneric(object_points, image_points, intrinsics, distortion,
                            ippe_rvecs, ippe_tvecs, false,
                            cv::SOLVEPNP_IPPE) == 2);
  CHECK(ippe_rvecs.size() == 2);

  std::vector<MarkerDetection> detections;
  const std::array<std::size_t, 4> order = {3, 0, 2, 1};
  for (std::size_t index : order) {
    detections.push_back(
        {image_points[index].x, image_points[index].y, ids[index]});
  }

  const LocalizationPipeline pipeline(map_path, window_size, geometry);
  const LocalizationResult result =
      pipeline.localize(detections, intrinsics, distortion, world_to_camera,
                        distance);
  CHECK(result.status == LocalizationStatus::SUCCESS);
  CHECK(result.pose_valid);
  CHECK(result.lookup.map_window_row == map_row);
  CHECK(result.lookup.map_window_col == map_col);
  const cv::Vec3d recovered_position(
      result.camera_position_world.at<double>(0, 0),
      result.camera_position_world.at<double>(1, 0),
      result.camera_position_world.at<double>(2, 0));
  CHECK_NEAR(cv::norm(recovered_position - expected_camera_position), 0.0,
             0.12);
  const cv::Mat expected_camera_rotation = cv::Mat(world_to_camera.t());
  CHECK_NEAR(cv::norm(result.camera_rotation_world - expected_camera_rotation,
                      cv::NORM_L2),
             0.0, 0.08);
  CHECK(result.reprojection_error < 1.0);
}

void testOppositePlaneSideAndEdgeOnRejection() {
  constexpr int window_size = 2;
  const std::string map_path = fixturePath("unique_grid.json");
  const MarkerGrid grid = MarkerGrid::fromJson(map_path, window_size);
  const cv::Mat intrinsics = cameraMatrix(900.0, 900.0, 640.0, 360.0);
  const cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);

  CameraPlaneGeometry geometry;
  constexpr double distance = 2.0;
  geometry.rounding_tolerance = 0.15;
  const cv::Matx33d world_to_camera = rpyRotation({180.0, 0.0, 0.0});

  const int map_row = 2;
  const int map_col = 1;
  const cv::Point3f top_left = grid.cellToGlobal(map_row, map_col);
  const cv::Vec3d expected_camera_position(top_left.x, top_left.y,
                                           top_left.z + distance);
  const cv::Vec3d translation = -(world_to_camera * expected_camera_position);
  cv::Mat expected_rvec;
  cv::Rodrigues(cv::Mat(world_to_camera), expected_rvec);
  const cv::Mat expected_tvec = (cv::Mat_<double>(3, 1) << translation[0],
                                 translation[1], translation[2]);

  std::vector<cv::Point3f> object_points;
  std::vector<int> ids;
  for (int dr = 0; dr < window_size; ++dr) {
    for (int dc = 0; dc < window_size; ++dc) {
      object_points.push_back(grid.cellToGlobal(map_row + dr, map_col + dc));
      ids.push_back(grid.cells()[map_row + dr][map_col + dc]);
    }
  }
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, expected_rvec, expected_tvec, intrinsics,
                    distortion, image_points);

  std::vector<MarkerDetection> detections;
  const std::array<std::size_t, 4> order = {2, 1, 3, 0};
  for (std::size_t index : order) {
    detections.push_back(
        {image_points[index].x, image_points[index].y, ids[index]});
  }

  const LocalizationPipeline pipeline(map_path, window_size, geometry);
  const LocalizationResult result =
      pipeline.localize(detections, intrinsics, distortion, world_to_camera,
                        distance);
  CHECK(result.status == LocalizationStatus::SUCCESS);
  CHECK(result.pose_valid);
  CHECK(result.lookup.map_window_row == map_row);
  CHECK(result.lookup.map_window_col == map_col);
  const cv::Vec3d recovered_position(
      result.camera_position_world.at<double>(0, 0),
      result.camera_position_world.at<double>(1, 0),
      result.camera_position_world.at<double>(2, 0));
  CHECK_NEAR(cv::norm(recovered_position - expected_camera_position), 0.0,
             2e-3);
  const cv::Mat expected_camera_rotation = cv::Mat(world_to_camera.t());
  CHECK_NEAR(cv::norm(result.camera_rotation_world - expected_camera_rotation,
                      cv::NORM_L2),
             0.0, 1e-3);

  const CameraMapper edge_on_mapper(grid.cellSpacing(), geometry);
  const cv::Matx33d edge_on_rotation = rpyRotation({0.0, 90.0, 0.0});
  const GridMappingResult edge_on = edge_on_mapper.detectionsToGrid(
      {{640.0F, 360.0F, 0}}, intrinsics, distortion, edge_on_rotation,
      distance);
  CHECK(!edge_on.valid);
  CHECK(edge_on.message.find("edge-on") != std::string::npos);
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"quaternion frame math", testQuaternionFrameMath},
      {"map parsing and validation", testMapParsingAndValidation},
      {"short-range map parsing and validation",
       testShortRangeMapParsingAndValidation},
      {"decoder gate", testDecoderGate},
      {"short-range lookup and pose", testShortRangeLookupAndPose},
      {"bidirectional cross-grid ID assignment",
       testCrossGridIdAssignmentIsBidirectional},
      {"cross-grid ID conflict rejection",
       testCrossGridIdAssignmentRejectsIncompleteAndConflictingEvidence},
      {"centered rectangular grid coordinates",
       testCenteredRectangularGridCoordinates},
      {"marker observation freshness", testMarkerObservationFreshness},
      {"w=2 duplicate IDs and shuffled input",
       testWindowTwoWithDuplicateIdsAndShuffledInput},
      {"w=3 shuffled input", testWindowThreeWithShuffledInput},
      {"lookup failures and ambiguity", testLookupFailuresAndAmbiguity},
      {"90-degree normalization with noise",
       testNormalizationWithNinetyDegreeOrientationAndNoise},
      {"quantization rejection", testQuantizationRejectsOffGridDetection},
      {"grid ID bootstrap and same-frame localization",
       testGridIdAssignmentBootstrapsAndLocalizesSameFrame},
      {"grid ID assignment requires decoded window",
       testGridIdAssignmentNeedsACompleteDecodedWindow},
      {"grid ID propagation across camera motion",
       testGridIdAssignmentPropagatesAcrossCameraMotion},
      {"grid ID conflict and duplicate rejection",
       testGridIdAssignmentRejectsConflictsAndDuplicateCells},
      {"grid ID split-anchor rejection",
       testGridIdAssignmentRejectsSplitAnchorConsensus},
      {"grid ID decoded-window recovery",
       testDecodedWindowRecoversFromAStaleInferredAnchor},
      {"map-aligned sparse localization",
       testMapAlignedLocalizationDoesNotNeedAnotherWindow},
      {"synthetic full localization",
       testSyntheticFullLocalizationRecoversPose},
      {"center 2x2 AP3P selection",
       testCenterTwoByTwoAp3pUsesClosestWindow},
      {"near-fronto-parallel IPPE selection",
       testNearFrontoParallelIppeUsesKnownGeometry},
      {"opposite plane side and edge-on rejection",
       testOppositePlaneSideAndEdgeOnRejection},
  };

  int failures = 0;
  for (const auto &test : tests) {
    try {
      test.second();
      std::cout << "[PASS] " << test.first << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
    }
  }
  std::cout << tests.size() - static_cast<std::size_t>(failures) << '/'
            << tests.size() << " tests passed\n";
  return failures == 0 ? 0 : 1;
}
