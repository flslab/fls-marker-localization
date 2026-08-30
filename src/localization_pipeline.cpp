#include "localization_pipeline.h"

#include "pose_estimator.h"
#include "pose_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <set>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {

constexpr double kPi = 3.14159265358979323846;

bool isFinite(double value) { return std::isfinite(value); }

double circularGridPhase(const std::vector<double> &coordinates) {
  double sum_sin = 0.0;
  double sum_cos = 0.0;
  for (double coordinate : coordinates) {
    const double angle = 2.0 * kPi * coordinate;
    sum_sin += std::sin(angle);
    sum_cos += std::cos(angle);
  }
  if (std::hypot(sum_sin, sum_cos) < 1e-12) {
    return coordinates.empty()
               ? 0.0
               : coordinates.front() - std::round(coordinates.front());
  }
  return std::atan2(sum_sin, sum_cos) / (2.0 * kPi);
}

double median(std::vector<double> values) {
  const auto middle = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle, values.end());
  if (values.size() % 2 != 0) {
    return *middle;
  }
  const double upper = *middle;
  return 0.5 *
         (upper + *std::max_element(values.begin(), middle));
}


std::vector<GlobalMarker>
selectCenterTwoByTwoWindow(const std::vector<GlobalMarker> &markers,
                           cv::Size frame_size) {
  std::map<std::pair<int, int>, const GlobalMarker *> by_cell;
  for (const auto &marker : markers) {
    by_cell[{marker.relative_row, marker.relative_col}] = &marker;
  }

  const cv::Point2d frame_center(frame_size.width * 0.5,
                                 frame_size.height * 0.5);
  double best_distance = std::numeric_limits<double>::infinity();
  std::vector<GlobalMarker> best;
  for (const auto &[cell, top_left] : by_cell) {
    const auto top_right = by_cell.find({cell.first, cell.second + 1});
    const auto bottom_left = by_cell.find({cell.first + 1, cell.second});
    const auto bottom_right = by_cell.find({cell.first + 1, cell.second + 1});
    if (top_right == by_cell.end() || bottom_left == by_cell.end() ||
        bottom_right == by_cell.end()) {
      continue;
    }

    const std::array<const GlobalMarker *, 4> window = {
        top_left, top_right->second, bottom_left->second, bottom_right->second};
    cv::Point2d centroid;
    for (const GlobalMarker *marker : window) {
      centroid += cv::Point2d(marker->image_x, marker->image_y);
    }
    centroid *= 0.25;
    const double distance = cv::norm(centroid - frame_center);
    if (distance < best_distance) {
      best_distance = distance;
      best.clear();
      for (const GlobalMarker *marker : window) {
        best.push_back(*marker);
      }
    }
  }
  return best;
}

std::string coordinateKey(int row, int col) {
  return std::to_string(row) + "," + std::to_string(col);
}

} // namespace

bool isMarkerObservationEligible(bool visible, double last_seen_age,
                                 double max_marker_age) {
  if (!isFinite(last_seen_age) || last_seen_age < 0.0 ||
      !isFinite(max_marker_age) || max_marker_age < 0.0) {
    return false;
  }
  return max_marker_age == 0.0 ? visible
                               : last_seen_age <= max_marker_age + 1e-9;
}

const char *gridLookupStatusName(GridLookupStatus status) {
  switch (status) {
  case GridLookupStatus::INSUFFICIENT_MARKERS:
    return "insufficient_markers";
  case GridLookupStatus::NO_COMPLETE_WINDOW:
    return "no_complete_window";
  case GridLookupStatus::NO_MATCH:
    return "no_match";
  case GridLookupStatus::AMBIGUOUS:
    return "ambiguous";
  case GridLookupStatus::UNIQUE:
    return "unique";
  }
  return "unknown";
}

MarkerGrid MarkerGrid::fromJson(const std::string &filename, int window_size) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open marker grid: " + filename);
  }

  json data;
  try {
    file >> data;
  } catch (const json::exception &error) {
    throw std::runtime_error("failed to parse marker grid '" + filename +
                             "': " + error.what());
  }

  const std::vector<std::string> required = {"rows",  "cols",         "num_ids",
                                             "min_k", "cell_spacing", "grid"};
  for (const auto &key : required) {
    if (!data.contains(key)) {
      throw std::runtime_error("marker grid is missing required field '" + key +
                               "'");
    }
  }

  MarkerGrid result;
  try {
    result.rows_ = data.at("rows").get<int>();
    result.cols_ = data.at("cols").get<int>();
    result.num_ids_ = data.at("num_ids").get<int>();
    result.min_k_ = data.at("min_k").get<int>();
    result.cell_spacing_ = data.at("cell_spacing").get<float>();
    result.grid_ = data.at("grid").get<std::vector<std::vector<int>>>();
    const auto origin =
        data.value("grid_origin", std::vector<float>{0.0F, 0.0F, 0.0F});
    if (origin.size() != 3) {
      throw std::runtime_error("grid_origin must contain exactly 3 numbers");
    }
    result.grid_origin_ = {origin[0], origin[1], origin[2]};
  } catch (const json::exception &error) {
    throw std::runtime_error("invalid marker grid schema: " +
                             std::string(error.what()));
  }

  if (result.rows_ <= 0 || result.cols_ <= 0 || result.num_ids_ <= 0) {
    throw std::runtime_error("rows, cols, and num_ids must be positive");
  }
  if (!isFinite(result.cell_spacing_) || result.cell_spacing_ <= 0.0F) {
    throw std::runtime_error("cell_spacing must be a positive finite number");
  }
  if (static_cast<int>(result.grid_.size()) != result.rows_) {
    throw std::runtime_error("grid row count does not match rows metadata");
  }
  for (int row = 0; row < result.rows_; ++row) {
    if (static_cast<int>(result.grid_[row].size()) != result.cols_) {
      throw std::runtime_error(
          "grid must be rectangular and match cols metadata");
    }
    for (int id : result.grid_[row]) {
      if (id < 0 || id >= result.num_ids_) {
        throw std::runtime_error(
            "grid contains marker ID outside [0, num_ids)");
      }
    }
  }
  if (window_size < 2 || window_size > result.rows_ ||
      window_size > result.cols_) {
    throw std::runtime_error(
        "window_size must be at least 2 and fit inside the marker grid");
  }

  result.window_size_ = window_size;
  result.buildWindowIndex();
  return result;
}

void MarkerGrid::buildWindowIndex() {
  window_index_.clear();
  total_window_count_ = 0;
  unique_window_count_ = 0;
  max_window_occurrences_ = 0;

  for (int row = 0; row <= rows_ - window_size_; ++row) {
    for (int col = 0; col <= cols_ - window_size_; ++col) {
      std::vector<int> signature;
      signature.reserve(window_size_ * window_size_);
      for (int dr = 0; dr < window_size_; ++dr) {
        for (int dc = 0; dc < window_size_; ++dc) {
          signature.push_back(grid_[row + dr][col + dc]);
        }
      }
      window_index_[signature].push_back({row, col});
      ++total_window_count_;
    }
  }

  for (const auto &entry : window_index_) {
    const int occurrences = static_cast<int>(entry.second.size());
    if (occurrences == 1) {
      ++unique_window_count_;
    }
    max_window_occurrences_ = std::max(max_window_occurrences_, occurrences);
  }
}

cv::Point3f MarkerGrid::cellToGlobal(int row, int col) const {
  if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
    throw std::out_of_range("marker grid cell is out of bounds");
  }
  return {grid_origin_.x - static_cast<float>(row) * cell_spacing_,
          grid_origin_.y - static_cast<float>(col) * cell_spacing_,
          grid_origin_.z};
}

GridLookupResult
MarkerGrid::lookup(const std::vector<RelativeMarker> &observations) const {
  GridLookupResult result;
  result.required_marker_count = window_size_ * window_size_;

  std::map<std::pair<int, int>, const RelativeMarker *> by_cell;
  for (const auto &observation : observations) {
    if (!observation.accepted) {
      continue;
    }
    const auto cell = std::make_pair(observation.row, observation.col);
    if (!by_cell.emplace(cell, &observation).second) {
      result.accepted_marker_count = static_cast<int>(by_cell.size());
      result.status = GridLookupStatus::NO_MATCH;
      result.message = "multiple detections normalized to relative cell " +
                       coordinateKey(observation.row, observation.col);
      return result;
    }
  }
  result.accepted_marker_count = static_cast<int>(by_cell.size());
  if (result.accepted_marker_count < result.required_marker_count) {
    result.status = GridLookupStatus::INSUFFICIENT_MARKERS;
    result.message = "need at least " +
                     std::to_string(result.required_marker_count) +
                     " accepted markers for a " + std::to_string(window_size_) +
                     "x" + std::to_string(window_size_) + " window";
    return result;
  }

  std::set<std::pair<int, int>> possible_local_origins;
  for (const auto &entry : by_cell) {
    for (int dr = 0; dr < window_size_; ++dr) {
      for (int dc = 0; dc < window_size_; ++dc) {
        possible_local_origins.insert(
            {entry.first.first - dr, entry.first.second - dc});
      }
    }
  }

  struct Candidate {
    int row_offset = 0;
    int col_offset = 0;
    int relative_window_row = -1;
    int relative_window_col = -1;
    int map_window_row = -1;
    int map_window_col = -1;
    int score = 0;
    std::vector<int> signature;
  };
  std::map<std::pair<int, int>, Candidate> candidates;

  for (const auto &local_origin : possible_local_origins) {
    std::vector<int> signature;
    signature.reserve(result.required_marker_count);
    bool complete = true;
    for (int dr = 0; dr < window_size_ && complete; ++dr) {
      for (int dc = 0; dc < window_size_; ++dc) {
        const auto it =
            by_cell.find({local_origin.first + dr, local_origin.second + dc});
        if (it == by_cell.end()) {
          complete = false;
          break;
        }
        signature.push_back(it->second->id);
      }
    }
    if (!complete) {
      continue;
    }
    ++result.complete_window_count;

    const auto map_matches = window_index_.find(signature);
    if (map_matches == window_index_.end()) {
      continue;
    }
    for (const auto &map_origin : map_matches->second) {
      const int row_offset = map_origin.first - local_origin.first;
      const int col_offset = map_origin.second - local_origin.second;
      Candidate candidate;
      candidate.row_offset = row_offset;
      candidate.col_offset = col_offset;
      candidate.relative_window_row = local_origin.first;
      candidate.relative_window_col = local_origin.second;
      candidate.map_window_row = map_origin.first;
      candidate.map_window_col = map_origin.second;
      candidate.signature = signature;
      candidates.emplace(std::make_pair(row_offset, col_offset),
                         std::move(candidate));
    }
  }

  if (result.complete_window_count == 0) {
    result.status = GridLookupStatus::NO_COMPLETE_WINDOW;
    result.message = "accepted markers do not contain a complete " +
                     std::to_string(window_size_) + "x" +
                     std::to_string(window_size_) + " window";
    return result;
  }
  if (candidates.empty()) {
    result.status = GridLookupStatus::NO_MATCH;
    result.message = "no complete observed window occurs in the marker map";
    return result;
  }

  int best_score = -1;
  for (auto &entry : candidates) {
    auto &candidate = entry.second;
    for (const auto &observation_entry : by_cell) {
      const int map_row = observation_entry.first.first + candidate.row_offset;
      const int map_col = observation_entry.first.second + candidate.col_offset;
      if (map_row >= 0 && map_row < rows_ && map_col >= 0 && map_col < cols_ &&
          grid_[map_row][map_col] == observation_entry.second->id) {
        ++candidate.score;
      }
    }
    best_score = std::max(best_score, candidate.score);
  }
  result.best_match_count = best_score;

  std::vector<const Candidate *> best_candidates;
  for (const auto &entry : candidates) {
    if (entry.second.score == best_score) {
      best_candidates.push_back(&entry.second);
    }
  }
  result.candidate_count = static_cast<int>(best_candidates.size());
  if (best_score != result.accepted_marker_count) {
    result.status = GridLookupStatus::NO_MATCH;
    result.message = "best map translation matches only " +
                     std::to_string(best_score) + "/" +
                     std::to_string(result.accepted_marker_count) +
                     " accepted markers";
    return result;
  }
  if (best_candidates.size() != 1) {
    result.status = GridLookupStatus::AMBIGUOUS;
    result.message = std::to_string(best_candidates.size()) +
                     " map translations tie with " +
                     std::to_string(best_score) + " matching markers";
    return result;
  }

  const Candidate &chosen = *best_candidates.front();
  result.relative_window_row = chosen.relative_window_row;
  result.relative_window_col = chosen.relative_window_col;
  result.map_window_row = chosen.map_window_row;
  result.map_window_col = chosen.map_window_col;
  result.window_signature = chosen.signature;

  for (const auto &observation : observations) {
    if (!observation.accepted) {
      continue;
    }
    const int map_row = observation.row + chosen.row_offset;
    const int map_col = observation.col + chosen.col_offset;
    if (map_row < 0 || map_row >= rows_ || map_col < 0 || map_col >= cols_ ||
        grid_[map_row][map_col] != observation.id) {
      continue;
    }
    const cv::Point3f global = cellToGlobal(map_row, map_col);
    result.markers.push_back({observation.detection_index, observation.row,
                              observation.col, map_row, map_col, observation.id,
                              observation.image_x, observation.image_y,
                              global.x, global.y, global.z});
  }

  result.status = GridLookupStatus::UNIQUE;
  result.message = "unique map translation found";
  return result;
}

CameraMapper::CameraMapper(float cell_spacing, CameraPlaneGeometry geometry)
    : cell_spacing_(cell_spacing), geometry_(std::move(geometry)) {
  if (!isFinite(cell_spacing_) || cell_spacing_ <= 0.0F) {
    throw std::invalid_argument("cell spacing must be positive and finite");
  }
  if (!isFinite(geometry_.rounding_tolerance) ||
      geometry_.rounding_tolerance <= 0.0 ||
      geometry_.rounding_tolerance >= 0.5) {
    throw std::invalid_argument(
        "grid rounding tolerance must be between 0 and 0.5 cells");
  }
}

GridMappingResult CameraMapper::detectionsToGrid(
    const std::vector<MarkerDetection> &detections,
    const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
    const cv::Matx33d &grid_to_camera_rotation,
    double camera_to_plane_distance) const {
  GridMappingResult result;
  if (detections.empty()) {
    result.message = "no marker detections";
    return result;
  }
  if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
    result.message = "camera matrix must be 3x3";
    return result;
  }
  for (double value : grid_to_camera_rotation.val) {
    if (!isFinite(value)) {
      result.message = "grid-to-camera rotation must be finite";
      return result;
    }
  }
  if (!isFinite(camera_to_plane_distance) || camera_to_plane_distance <= 0.0) {
    result.message = "camera-to-plane distance must be positive and finite";
    return result;
  }

  std::vector<cv::Point2f> image_points;
  image_points.reserve(detections.size());
  for (const auto &detection : detections) {
    if (!isFinite(detection.x) || !isFinite(detection.y)) {
      result.message = "detections must have finite coordinates";
      return result;
    }
    image_points.emplace_back(detection.x, detection.y);
  }

  std::vector<cv::Point2f> plane_points;
  try {
    std::vector<cv::Point2f> normalized_points;
    cv::undistortPoints(image_points, normalized_points, camera_matrix,
                        dist_coeffs);

    const cv::Mat rotation = cv::Mat(grid_to_camera_rotation);
    const double normal_depth = rotation.at<double>(2, 2);
    if (std::abs(normal_depth) < 1e-4) {
      result.message =
          "marker plane is edge-on to the camera; normalization is unstable";
      return result;
    }

    cv::Mat homography(3, 3, CV_64F);
    rotation.col(0).copyTo(homography.col(0));
    rotation.col(1).copyTo(homography.col(1));
    rotation.col(2).copyTo(homography.col(2));
    // Distance is unsigned. Choose the camera-to-plane normal branch whose
    // perpendicular foot has positive OpenCV camera-Z (is in front).
    const double signed_distance =
        std::copysign(camera_to_plane_distance, normal_depth);
    homography.col(2) *= signed_distance;
    const cv::Mat inverse_homography = homography.inv();

    plane_points.reserve(normalized_points.size());
    for (const auto &point : normalized_points) {
      const cv::Mat ray = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
      const cv::Mat plane_homogeneous = inverse_homography * ray;
      const double scale = plane_homogeneous.at<double>(2, 0);
      if (!isFinite(scale) || std::abs(scale) < 1e-9) {
        result.message = "camera ray is parallel to the marker plane";
        return result;
      }
      plane_points.emplace_back(
          static_cast<float>(plane_homogeneous.at<double>(0, 0) / scale),
          static_cast<float>(plane_homogeneous.at<double>(1, 0) / scale));
    }
  } catch (const cv::Exception &error) {
    result.message =
        "failed to rectify marker detections: " + std::string(error.what());
    return result;
  }

  std::vector<double> row_coordinates;
  std::vector<double> col_coordinates;
  row_coordinates.reserve(plane_points.size());
  col_coordinates.reserve(plane_points.size());
  for (const auto &point : plane_points) {
    row_coordinates.push_back(-point.x / cell_spacing_);
    col_coordinates.push_back(-point.y / cell_spacing_);
  }
  const double row_phase = circularGridPhase(row_coordinates);
  const double col_phase = circularGridPhase(col_coordinates);

  std::vector<int> raw_rows;
  std::vector<int> raw_cols;
  raw_rows.reserve(detections.size());
  raw_cols.reserve(detections.size());
  int min_row = std::numeric_limits<int>::max();
  int min_col = std::numeric_limits<int>::max();
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const int row =
        static_cast<int>(std::llround(row_coordinates[i] - row_phase));
    const int col =
        static_cast<int>(std::llround(col_coordinates[i] - col_phase));
    raw_rows.push_back(row);
    raw_cols.push_back(col);
    min_row = std::min(min_row, row);
    min_col = std::min(min_col, col);
  }

  result.markers.reserve(detections.size());
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const double row_nearest = std::round(row_coordinates[i] - row_phase);
    const double col_nearest = std::round(col_coordinates[i] - col_phase);
    const float row_error = static_cast<float>(
        std::abs((row_coordinates[i] - row_phase) - row_nearest));
    const float col_error = static_cast<float>(
        std::abs((col_coordinates[i] - col_phase) - col_nearest));
    const bool accepted = row_error <= geometry_.rounding_tolerance &&
                          col_error <= geometry_.rounding_tolerance;
    result.markers.push_back({i, detections[i].x, detections[i].y,
                              detections[i].id, raw_rows[i] - min_row,
                              raw_cols[i] - min_col,
                              static_cast<float>(row_coordinates[i]),
                              static_cast<float>(col_coordinates[i]), row_error,
                              col_error, accepted});
  }

  result.valid = true;
  result.message = "detections rectified and normalized to grid cells";
  return result;
}

GridIdAssigner::GridIdAssigner(const MarkerGrid &grid,
                               CameraPlaneGeometry geometry)
    : grid_(grid), geometry_(std::move(geometry)),
      camera_mapper_(grid.cellSpacing(), geometry_) {}

void GridIdAssigner::forgetTracks(
    const std::vector<std::uint64_t> &track_ids) {
  for (const std::uint64_t track_id : track_ids) {
    track_cells_.erase(track_id);
  }
}

GridIdAssignmentResult GridIdAssigner::assign(
    const std::vector<MarkerDetection> &decoded_detections,
    const std::vector<MarkerDetection> &current_blobs,
    const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
    const cv::Matx33d &grid_to_camera_rotation,
    double camera_to_plane_distance) {
  GridIdAssignmentResult result;

  // Only decoder-confirmed, currently visible observations may establish a
  // map lock. Unknown blobs never participate in the initial lattice phase.
  std::vector<MarkerDetection> visible_decoded;
  for (const auto &detection : decoded_detections) {
    if (detection.visible && detection.id >= 0) {
      visible_decoded.push_back(detection);
    }
  }
  if (!visible_decoded.empty()) {
    const GridMappingResult decoded_mapping = camera_mapper_.detectionsToGrid(
        visible_decoded, camera_matrix, dist_coeffs, grid_to_camera_rotation,
        camera_to_plane_distance);
    if (decoded_mapping.valid) {
      const GridLookupResult lookup = grid_.lookup(decoded_mapping.markers);
      if (lookup.status == GridLookupStatus::UNIQUE) {
        bool overlaps_lock = false;
        bool conflicts_with_lock = false;
        for (const auto &marker : lookup.markers) {
          if (marker.detection_index >= visible_decoded.size()) {
            continue;
          }
          const auto known = track_cells_.find(
              visible_decoded[marker.detection_index].track_id);
          if (known == track_cells_.end()) {
            continue;
          }
          if (known->second.row != marker.map_row ||
              known->second.col != marker.map_col) {
            conflicts_with_lock = true;
          } else {
            overlaps_lock = true;
          }
        }

        int visible_decoder_support = 0;
        for (const auto &detection : current_blobs) {
          const auto known = track_cells_.find(detection.track_id);
          if (known != track_cells_.end() && detection.id >= 0 &&
              detection.id ==
                  grid_.cells()[known->second.row][known->second.col]) {
            ++visible_decoder_support;
          }
        }
        const bool fresh_window_is_stronger =
            static_cast<int>(lookup.markers.size()) > visible_decoder_support;
        const bool replace_lock =
            fresh_window_is_stronger &&
            (!overlaps_lock || conflicts_with_lock);
        if ((!conflicts_with_lock || replace_lock) &&
            (track_cells_.empty() || overlaps_lock || replace_lock)) {
          if (replace_lock) {
            track_cells_.clear();
          }
          for (const auto &marker : lookup.markers) {
            if (marker.detection_index >= visible_decoded.size()) {
              continue;
            }
            const std::uint64_t track_id =
                visible_decoded[marker.detection_index].track_id;
            if (track_id != 0) {
              track_cells_[track_id] = {marker.map_row, marker.map_col};
            }
          }
        }
      }
    }
  }
  result.map_locked = !track_cells_.empty();

  const auto decoder_fallback = [&]() {
    for (const auto &detection : decoded_detections) {
      const auto known = track_cells_.find(detection.track_id);
      if (known == track_cells_.end()) {
        if (!result.map_locked) {
          result.detections.push_back(detection);
        }
        continue;
      }
      if (detection.id !=
          grid_.cells()[known->second.row][known->second.col]) {
        continue;
      }
      MarkerDetection mapped = detection;
      mapped.map_row = known->second.row;
      mapped.map_col = known->second.col;
      result.detections.push_back(mapped);
    }
    result.message = result.map_locked
                         ? "map locked; waiting for a visible assigned anchor"
                         : "waiting for a unique decoded map window";
  };
  if (!result.map_locked || current_blobs.empty()) {
    decoder_fallback();
    return result;
  }

  const GridMappingResult current_mapping = camera_mapper_.detectionsToGrid(
      current_blobs, camera_matrix, dist_coeffs, grid_to_camera_rotation,
      camera_to_plane_distance);
  if (!current_mapping.valid) {
    decoder_fallback();
    result.message = current_mapping.message;
    return result;
  }

  struct AnchorOffset {
    double row = 0.0;
    double col = 0.0;
  };
  std::vector<AnchorOffset> offsets;
  for (std::size_t index = 0; index < current_blobs.size(); ++index) {
    const auto known = track_cells_.find(current_blobs[index].track_id);
    if (known == track_cells_.end()) {
      continue;
    }
    const Cell &cell = known->second;
    if (current_blobs[index].id >= 0 &&
        current_blobs[index].id != grid_.cells()[cell.row][cell.col]) {
      continue;
    }
    offsets.push_back(
        {current_mapping.markers[index].row_coordinate - cell.row,
         current_mapping.markers[index].col_coordinate - cell.col});
  }
  if (offsets.empty()) {
    decoder_fallback();
    return result;
  }

  // Find the largest mutually compatible anchor group before taking its
  // median. A bad tracker association cannot move a stronger lattice cluster.
  std::vector<std::size_t> best_consensus;
  bool ambiguous_consensus = false;
  double best_error = std::numeric_limits<double>::infinity();
  const double consensus_tolerance = 2.0 * geometry_.rounding_tolerance;
  for (const AnchorOffset &hypothesis : offsets) {
    std::vector<std::size_t> consensus;
    double error = 0.0;
    for (std::size_t index = 0; index < offsets.size(); ++index) {
      const double row_error = std::abs(offsets[index].row - hypothesis.row);
      const double col_error = std::abs(offsets[index].col - hypothesis.col);
      if (row_error <= consensus_tolerance &&
          col_error <= consensus_tolerance) {
        consensus.push_back(index);
        error += row_error + col_error;
      }
    }
    if (consensus.size() > best_consensus.size()) {
      best_consensus = std::move(consensus);
      best_error = error;
      ambiguous_consensus = false;
    } else if (consensus.size() == best_consensus.size()) {
      const bool disjoint = std::none_of(
          consensus.begin(), consensus.end(), [&](std::size_t index) {
            return std::find(best_consensus.begin(), best_consensus.end(),
                             index) != best_consensus.end();
          });
      ambiguous_consensus = ambiguous_consensus || disjoint;
      if (error < best_error) {
        best_consensus = std::move(consensus);
        best_error = error;
      }
    }
  }
  if (ambiguous_consensus ||
      (offsets.size() > 1 && best_consensus.size() == 1)) {
    decoder_fallback();
    result.message = "visible assigned anchors disagree on lattice alignment";
    return result;
  }

  std::vector<double> row_offsets;
  std::vector<double> col_offsets;
  for (std::size_t index : best_consensus) {
    row_offsets.push_back(offsets[index].row);
    col_offsets.push_back(offsets[index].col);
  }
  const double row_offset = median(std::move(row_offsets));
  const double col_offset = median(std::move(col_offsets));
  const double anchor_tolerance = geometry_.rounding_tolerance + 1e-9;
  const int aligned_anchor_count = static_cast<int>(std::count_if(
      offsets.begin(), offsets.end(), [&](const AnchorOffset &offset) {
        return std::abs(offset.row - row_offset) <= anchor_tolerance &&
               std::abs(offset.col - col_offset) <= anchor_tolerance;
      }));
  if (offsets.size() > 1 && aligned_anchor_count < 2) {
    decoder_fallback();
    result.message = "visible assigned anchors disagree on lattice alignment";
    return result;
  }
  result.alignment_valid = true;

  struct Candidate {
    int row = -1;
    int col = -1;
    bool valid = false;
    bool known = false;
    bool decoded = false;
  };
  std::vector<Candidate> candidates(current_blobs.size());
  const double tolerance = geometry_.rounding_tolerance + 1e-9;
  for (std::size_t index = 0; index < current_blobs.size(); ++index) {
    const double row_value =
        current_mapping.markers[index].row_coordinate - row_offset;
    const double col_value =
        current_mapping.markers[index].col_coordinate - col_offset;
    if (!isFinite(row_value) || !isFinite(col_value) ||
        row_value < std::numeric_limits<int>::min() ||
        row_value > std::numeric_limits<int>::max() ||
        col_value < std::numeric_limits<int>::min() ||
        col_value > std::numeric_limits<int>::max()) {
      continue;
    }

    Candidate &candidate = candidates[index];
    candidate.row = static_cast<int>(std::llround(row_value));
    candidate.col = static_cast<int>(std::llround(col_value));
    candidate.decoded = current_blobs[index].id >= 0;
    if (std::abs(row_value - candidate.row) > tolerance ||
        std::abs(col_value - candidate.col) > tolerance ||
        candidate.row < 0 || candidate.row >= grid_.rows() ||
        candidate.col < 0 || candidate.col >= grid_.cols()) {
      continue;
    }

    const auto known = track_cells_.find(current_blobs[index].track_id);
    candidate.known = known != track_cells_.end();
    if (candidate.known &&
        (known->second.row != candidate.row ||
         known->second.col != candidate.col)) {
      continue;
    }
    if (candidate.decoded &&
        current_blobs[index].id !=
            grid_.cells()[candidate.row][candidate.col]) {
      continue;
    }
    candidate.valid = true;
  }

  std::vector<bool> accepted(current_blobs.size(), false);
  std::set<std::pair<int, int>> occupied_cells;
  std::map<std::pair<int, int>, std::vector<std::size_t>> known_claims;
  std::map<std::pair<int, int>, std::vector<std::size_t>> new_claims;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (!candidates[index].valid) {
      continue;
    }
    const auto cell =
        std::make_pair(candidates[index].row, candidates[index].col);
    (candidates[index].known ? known_claims : new_claims)[cell].push_back(index);
  }

  // A cached track owns its cell. Multiple cached owners are ambiguous and
  // therefore all rejected for this frame.
  std::map<std::pair<int, int>, std::size_t> known_owners;
  for (const auto &[cell, claims] : known_claims) {
    if (claims.size() == 1) {
      accepted[claims.front()] = true;
      occupied_cells.insert(cell);
      known_owners[cell] = claims.front();
    }
  }

  std::vector<std::size_t> pending_inferred;
  for (const auto &[cell, claims] : new_claims) {
    std::vector<std::size_t> decoded_claims;
    for (std::size_t index : claims) {
      if (candidates[index].decoded) {
        decoded_claims.push_back(index);
      }
    }
    if (occupied_cells.count(cell) != 0) {
      const auto known_owner = known_owners.find(cell);
      if (decoded_claims.size() == 1 && known_owner != known_owners.end() &&
          !candidates[known_owner->second].decoded) {
        accepted[known_owner->second] = false;
        track_cells_.erase(current_blobs[known_owner->second].track_id);
        const std::size_t index = decoded_claims.front();
        accepted[index] = true;
        track_cells_[current_blobs[index].track_id] =
            {candidates[index].row, candidates[index].col};
      } else if (!decoded_claims.empty() &&
                 known_owner != known_owners.end()) {
        accepted[known_owner->second] = false;
        occupied_cells.erase(cell);
      }
      continue;
    }
    if (decoded_claims.size() == 1) {
      const std::size_t index = decoded_claims.front();
      accepted[index] = true;
      occupied_cells.insert(cell);
      track_cells_[current_blobs[index].track_id] =
          {candidates[index].row, candidates[index].col};
    } else if (decoded_claims.empty() && claims.size() == 1) {
      pending_inferred.push_back(claims.front());
    }
  }

  // Infer one frontier layer per frame. Chebyshev adjacency includes diagonal
  // neighbors while preventing an on-lattice artifact chain from growing an
  // arbitrary distance in one observation.
  const std::set<std::pair<int, int>> frontier_cells = occupied_cells;
  for (std::size_t index : pending_inferred) {
    const Candidate &candidate = candidates[index];
    const bool adjacent = std::any_of(
        frontier_cells.begin(), frontier_cells.end(), [&](const auto &cell) {
          return std::max(std::abs(cell.first - candidate.row),
                          std::abs(cell.second - candidate.col)) == 1;
        });
    if (adjacent) {
      accepted[index] = true;
      track_cells_[current_blobs[index].track_id] = {candidate.row,
                                                     candidate.col};
    }
  }

  std::set<std::uint64_t> current_track_ids;
  for (std::size_t index = 0; index < current_blobs.size(); ++index) {
    current_track_ids.insert(current_blobs[index].track_id);
    if (!accepted[index]) {
      continue;
    }
    MarkerDetection detection = current_blobs[index];
    detection.inferred = detection.id < 0;
    detection.id = grid_.cells()[candidates[index].row][candidates[index].col];
    detection.map_row = candidates[index].row;
    detection.map_col = candidates[index].col;
    result.inferred_marker_count += detection.inferred ? 1 : 0;
    result.detections.push_back(detection);
  }
  result.rejected_blob_count =
      static_cast<int>(current_blobs.size() - result.detections.size());

  // Retain explicitly enabled, recently seen decoder observations only when
  // their map cell is already known. Current-frame conflicts stay rejected.
  for (const auto &detection : decoded_detections) {
    if (current_track_ids.count(detection.track_id) != 0) {
      continue;
    }
    const auto known = track_cells_.find(detection.track_id);
    if (known == track_cells_.end() ||
        detection.id != grid_.cells()[known->second.row][known->second.col]) {
      continue;
    }
    MarkerDetection mapped = detection;
    mapped.map_row = known->second.row;
    mapped.map_col = known->second.col;
    result.detections.push_back(mapped);
  }

  result.message = "map IDs assigned from visible lattice continuity";
  return result;
}

const char *localizationStatusName(LocalizationStatus status) {
  switch (status) {
  case LocalizationStatus::NO_DETECTIONS:
    return "no_detections";
  case LocalizationStatus::INSUFFICIENT_MARKERS:
    return "insufficient_markers";
  case LocalizationStatus::NORMALIZATION_FAILED:
    return "normalization_failed";
  case LocalizationStatus::NO_COMPLETE_WINDOW:
    return "no_complete_window";
  case LocalizationStatus::NO_MAP_MATCH:
    return "no_map_match";
  case LocalizationStatus::AMBIGUOUS_MAP_MATCH:
    return "ambiguous_map_match";
  case LocalizationStatus::PNP_FAILED:
    return "pnp_failed";
  case LocalizationStatus::SUCCESS:
    return "success";
  }
  return "unknown";
}

LocalizationPipeline::LocalizationPipeline(const std::string &map_file,
                                           int window_size,
                                           CameraPlaneGeometry geometry,
                                           bool center_window_ap3p)
    : grid_(MarkerGrid::fromJson(map_file, window_size)),
      geometry_(std::move(geometry)),
      camera_mapper_(grid_.cellSpacing(), geometry_),
      center_window_ap3p_(center_window_ap3p) {
  if (center_window_ap3p_ && window_size != 2) {
    throw std::invalid_argument("center-window AP3P requires window_size 2");
  }
}

LocalizationResult LocalizationPipeline::localize(
    const std::vector<MarkerDetection> &detections,
    const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
    const cv::Matx33d &grid_to_camera_rotation, double distance,
    cv::Size frame_size) const {
  LocalizationResult result;
  result.distance_used = distance;
  result.pnp_solver = center_window_ap3p_ ? "ap3p" : "ippe_iterative";
  result.lookup.required_marker_count = grid_.windowSize() * grid_.windowSize();
  if (detections.empty()) {
    result.status = LocalizationStatus::NO_DETECTIONS;
    result.message = "no marker detections";
    return result;
  }

  const GridMappingResult mapping =
      camera_mapper_.detectionsToGrid(detections, camera_matrix, dist_coeffs,
                                      grid_to_camera_rotation, distance);
  result.relative_markers = mapping.markers;
  if (!mapping.valid) {
    result.status = LocalizationStatus::NORMALIZATION_FAILED;
    result.message = mapping.message;
    return result;
  }

  const int accepted_count = static_cast<int>(std::count_if(
      mapping.markers.begin(), mapping.markers.end(),
      [](const RelativeMarker &marker) { return marker.accepted; }));
  const bool map_aligned =
      std::all_of(detections.begin(), detections.end(),
                  [](const MarkerDetection &detection) {
                    return detection.hasMapCell();
                  });
  if (map_aligned) {
    result.lookup.required_marker_count = 4;
    result.lookup.accepted_marker_count = accepted_count;
    std::set<std::pair<int, int>> map_cells;
    int row_translation = 0;
    int col_translation = 0;
    bool have_translation = false;
    for (const auto &marker : mapping.markers) {
      if (!marker.accepted || marker.detection_index >= detections.size()) {
        continue;
      }
      const MarkerDetection &detection =
          detections[marker.detection_index];
      if (detection.map_row >= grid_.rows() ||
          detection.map_col >= grid_.cols() ||
          grid_.cells()[detection.map_row][detection.map_col] != detection.id) {
        result.lookup.status = GridLookupStatus::NO_MATCH;
        result.status = LocalizationStatus::NO_MAP_MATCH;
        result.message = "assigned marker conflicts with the marker map";
        result.lookup.message = result.message;
        return result;
      }
      if (!map_cells.emplace(detection.map_row, detection.map_col).second) {
        result.lookup.status = GridLookupStatus::NO_MATCH;
        result.status = LocalizationStatus::NO_MAP_MATCH;
        result.message = "multiple detections claim one assigned map cell";
        result.lookup.message = result.message;
        return result;
      }

      const int candidate_row_translation = detection.map_row - marker.row;
      const int candidate_col_translation = detection.map_col - marker.col;
      if (have_translation &&
          (candidate_row_translation != row_translation ||
           candidate_col_translation != col_translation)) {
        result.lookup.status = GridLookupStatus::NO_MATCH;
        result.status = LocalizationStatus::NO_MAP_MATCH;
        result.message = "assigned map cells do not fit one lattice";
        result.lookup.message = result.message;
        return result;
      }
      row_translation = candidate_row_translation;
      col_translation = candidate_col_translation;
      have_translation = true;

      const cv::Point3f global =
          grid_.cellToGlobal(detection.map_row, detection.map_col);
      result.lookup.markers.push_back(
          {marker.detection_index, marker.row, marker.col, detection.map_row,
           detection.map_col, detection.id, detection.x, detection.y,
           global.x, global.y, global.z});
    }
    result.lookup.accepted_marker_count =
        static_cast<int>(result.lookup.markers.size());
    result.lookup.best_match_count = result.lookup.accepted_marker_count;
    result.lookup.candidate_count = result.lookup.markers.empty() ? 0 : 1;
    if (result.lookup.markers.size() < 4) {
      result.lookup.status = GridLookupStatus::INSUFFICIENT_MARKERS;
      result.status = LocalizationStatus::INSUFFICIENT_MARKERS;
      result.message = "PnP needs at least four assigned grid markers";
      result.lookup.message = result.message;
      return result;
    }
    result.lookup.status = GridLookupStatus::UNIQUE;
    result.lookup.message = "using map-aligned marker assignments";
  } else {
    if (accepted_count < result.lookup.required_marker_count) {
      result.lookup = grid_.lookup(mapping.markers);
      result.status = LocalizationStatus::INSUFFICIENT_MARKERS;
      result.message = result.lookup.message;
      return result;
    }

    result.lookup = grid_.lookup(mapping.markers);
    switch (result.lookup.status) {
    case GridLookupStatus::INSUFFICIENT_MARKERS:
      result.status = LocalizationStatus::INSUFFICIENT_MARKERS;
      result.message = result.lookup.message;
      return result;
    case GridLookupStatus::NO_COMPLETE_WINDOW:
      result.status = LocalizationStatus::NO_COMPLETE_WINDOW;
      result.message = result.lookup.message;
      return result;
    case GridLookupStatus::NO_MATCH:
      result.status = LocalizationStatus::NO_MAP_MATCH;
      result.message = result.lookup.message;
      return result;
    case GridLookupStatus::AMBIGUOUS:
      result.status = LocalizationStatus::AMBIGUOUS_MAP_MATCH;
      result.message = result.lookup.message;
      return result;
    case GridLookupStatus::UNIQUE:
      break;
    }
  }

  if (result.lookup.markers.size() < 4) {
    result.status = LocalizationStatus::PNP_FAILED;
    result.message = "PnP needs at least four matched grid markers";
    return result;
  }

  if (center_window_ap3p_) {
    if (frame_size.width <= 0 || frame_size.height <= 0) {
      result.status = LocalizationStatus::PNP_FAILED;
      result.message = "center-window AP3P needs the image dimensions";
      return result;
    }
    result.pose_markers =
        selectCenterTwoByTwoWindow(result.lookup.markers, frame_size);
    if (result.pose_markers.size() != 4) {
      result.status = LocalizationStatus::PNP_FAILED;
      result.message = "no complete 2x2 grid window is available for AP3P";
      return result;
    }
  } else {
    result.pose_markers = result.lookup.markers;
  }

  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> object_points;
  image_points.reserve(result.pose_markers.size());
  object_points.reserve(result.pose_markers.size());
  for (const auto &marker : result.pose_markers) {
    image_points.emplace_back(marker.image_x, marker.image_y);
    object_points.emplace_back(marker.global_x, marker.global_y,
                               marker.global_z);
  }

  pose_estimation::PlanarPosePrior prior;
  prior.distance = distance;
  prior.expected_object_to_camera_rotation =
      cv::Mat(grid_to_camera_rotation).clone();
  pose_estimation::PnpEstimate pose;
  if (center_window_ap3p_) {
    pose = pose_estimation::solveAp3p(object_points, image_points,
                                      camera_matrix, dist_coeffs, prior);
  } else {
    pose = pose_estimation::solvePlanarIppe(
        object_points, image_points, camera_matrix, dist_coeffs, prior);
  }
  if (!pose.valid) {
    result.status = LocalizationStatus::PNP_FAILED;
    result.message = pose.message;
    return result;
  }

  result.tvec_world_to_camera = pose.object_to_camera_translation;
  cv::Rodrigues(cv::Mat(grid_to_camera_rotation),
                result.rvec_world_to_camera);
  result.camera_rotation_world = cv::Mat(grid_to_camera_rotation.t()).clone();
  result.camera_position_world =
      -result.camera_rotation_world * result.tvec_world_to_camera;
  result.camera_roll_pitch_yaw =
      pose_math::rpyFromRotation(result.camera_rotation_world);
  const double plane_z = object_points.front().z;
  result.camera_to_plane_distance = std::abs(
      result.camera_position_world.at<double>(2, 0) - plane_z);
  if (!isFinite(result.camera_to_plane_distance) ||
      result.camera_to_plane_distance <= 0.0) {
    result.status = LocalizationStatus::PNP_FAILED;
    result.message = "shared attitude produced an invalid plane distance";
    return result;
  }
  result.reprojection_error = pose.rms_reprojection_error;
  result.pose_valid = true;

  result.status = LocalizationStatus::SUCCESS;
  result.message = "global camera pose solved from marker grid";
  return result;
}
