#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct MarkerDetection {
  float x = 0.0F;
  float y = 0.0F;
  int id = -1;
  std::uint64_t track_id = 0;
  bool visible = true;
  double last_seen_age = 0.0;
  bool inferred = false;
  int map_row = -1;
  int map_col = -1;

  bool hasMapCell() const { return map_row >= 0 && map_col >= 0; }
};

// A zero age limit means strictly current-frame/visible observations. Positive
// limits permit recently seen coordinates for stationary-camera deployments.
bool isMarkerObservationEligible(bool visible, double last_seen_age,
                                 double max_marker_age);

struct RelativeMarker {
  std::size_t detection_index = 0;
  float image_x = 0.0F;
  float image_y = 0.0F;
  int id = -1;
  int row = 0;
  int col = 0;
  float row_coordinate = 0.0F;
  float col_coordinate = 0.0F;
  float row_rounding_error = 0.0F;
  float col_rounding_error = 0.0F;
  bool accepted = false;
};

struct GlobalMarker {
  std::size_t detection_index = 0;
  int relative_row = 0;
  int relative_col = 0;
  int map_row = 0;
  int map_col = 0;
  int id = -1;
  float image_x = 0.0F;
  float image_y = 0.0F;
  float global_x = 0.0F;
  float global_y = 0.0F;
  float global_z = 0.0F;
};

enum class GridLookupStatus {
  INSUFFICIENT_MARKERS,
  NO_COMPLETE_WINDOW,
  NO_MATCH,
  AMBIGUOUS,
  UNIQUE,
};

const char *gridLookupStatusName(GridLookupStatus status);

struct GridLookupResult {
  GridLookupStatus status = GridLookupStatus::INSUFFICIENT_MARKERS;
  std::string message;
  int required_marker_count = 0;
  int accepted_marker_count = 0;
  int complete_window_count = 0;
  int candidate_count = 0;
  int best_match_count = 0;
  int relative_window_row = -1;
  int relative_window_col = -1;
  int map_window_row = -1;
  int map_window_col = -1;
  std::vector<int> window_signature;
  std::vector<GlobalMarker> markers;
};

class MarkerGrid {
public:
  static MarkerGrid fromJson(const std::string &filename, int window_size);

  GridLookupResult
  lookup(const std::vector<RelativeMarker> &observations) const;

  cv::Point3f cellToGlobal(int row, int col) const;

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int numIds() const { return num_ids_; }
  int minK() const { return min_k_; }
  int windowSize() const { return window_size_; }
  int totalWindowCount() const { return total_window_count_; }
  int uniqueWindowCount() const { return unique_window_count_; }
  int maxWindowOccurrences() const { return max_window_occurrences_; }
  float cellSpacing() const { return cell_spacing_; }
  // World coordinate of grid cell (0, 0), not the geometric grid centre.
  const cv::Point3f &gridOrigin() const { return grid_origin_; }
  const std::vector<std::vector<int>> &cells() const { return grid_; }

private:
  int rows_ = 0;
  int cols_ = 0;
  int num_ids_ = 0;
  int min_k_ = 0;
  int window_size_ = 0;
  int total_window_count_ = 0;
  int unique_window_count_ = 0;
  int max_window_occurrences_ = 0;
  float cell_spacing_ = 0.0F;
  cv::Point3f grid_origin_{0.0F, 0.0F, 0.0F};
  std::vector<std::vector<int>> grid_;
  std::map<std::vector<int>, std::vector<std::pair<int, int>>> window_index_;

  void buildWindowIndex();
};

struct CameraPlaneGeometry {
  // Maximum distance from a rectified coordinate to its nearest grid line,
  // expressed in grid cells.
  double rounding_tolerance = 0.30;
};

struct GridMappingResult {
  bool valid = false;
  std::string message;
  std::vector<RelativeMarker> markers;
};

class CameraMapper {
public:
  CameraMapper(float cell_spacing, CameraPlaneGeometry geometry);

  GridMappingResult
  detectionsToGrid(const std::vector<MarkerDetection> &detections,
                   const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
                   const cv::Matx33d &grid_to_camera_rotation,
                   double camera_to_plane_distance) const;

private:
  float cell_spacing_;
  CameraPlaneGeometry geometry_;
};

struct GridIdAssignmentResult {
  bool map_locked = false;
  bool alignment_valid = false;
  int inferred_marker_count = 0;
  int rejected_blob_count = 0;
  std::string message;
  std::vector<MarkerDetection> detections;
};

// Assigns map IDs from spatial continuity after a decoded window establishes
// the map location. It deliberately does not modify MarkerTracker state.
class GridIdAssigner {
public:
  GridIdAssigner(const MarkerGrid &grid, CameraPlaneGeometry geometry);

  GridIdAssignmentResult assign(
      const std::vector<MarkerDetection> &decoded_detections,
      const std::vector<MarkerDetection> &current_blobs,
      const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
      const cv::Matx33d &grid_to_camera_rotation,
      double camera_to_plane_distance);

  void forgetTracks(const std::vector<std::uint64_t> &track_ids);
  bool mapLocked() const { return !track_cells_.empty(); }

private:
  struct Cell {
    int row = -1;
    int col = -1;
  };

  const MarkerGrid &grid_;
  CameraPlaneGeometry geometry_;
  CameraMapper camera_mapper_;
  std::map<std::uint64_t, Cell> track_cells_;
};

enum class LocalizationStatus {
  NO_DETECTIONS,
  INSUFFICIENT_MARKERS,
  NORMALIZATION_FAILED,
  NO_COMPLETE_WINDOW,
  NO_MAP_MATCH,
  AMBIGUOUS_MAP_MATCH,
  PNP_FAILED,
  SUCCESS,
};

const char *localizationStatusName(LocalizationStatus status);

struct LocalizationResult {
  LocalizationStatus status = LocalizationStatus::NO_DETECTIONS;
  std::string message;
  std::vector<RelativeMarker> relative_markers;
  GridLookupResult lookup;
  std::vector<GlobalMarker> pose_markers;
  std::string pnp_solver;

  bool pose_valid = false;
  // Shared-attitude rotation with the PnP translation:
  // X_camera = R_shared * X_world + t_pnp.
  cv::Mat rvec_world_to_camera;
  cv::Mat tvec_world_to_camera;
  cv::Mat camera_rotation_world;
  cv::Mat camera_position_world;
  cv::Vec3d camera_roll_pitch_yaw{0.0, 0.0, 0.0};
  double reprojection_error = -1.0;
  double distance_used = -1.0;
  double camera_to_plane_distance = -1.0;
};

class LocalizationPipeline {
public:
  LocalizationPipeline(const std::string &map_file, int window_size,
                       CameraPlaneGeometry geometry,
                       bool center_window_ap3p = false);

  LocalizationResult localize(
      const std::vector<MarkerDetection> &detections,
      const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
      const cv::Matx33d &grid_to_camera_rotation, double distance,
      cv::Size frame_size = {}) const;

  const MarkerGrid &grid() const { return grid_; }
  const CameraPlaneGeometry &geometry() const { return geometry_; }

private:
  MarkerGrid grid_;
  CameraPlaneGeometry geometry_;
  CameraMapper camera_mapper_;
  bool center_window_ap3p_ = false;
};
