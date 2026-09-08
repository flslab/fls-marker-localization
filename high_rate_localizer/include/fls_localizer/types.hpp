#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace flsloc {

enum class LocalizerState : std::uint8_t {
  Starting = 0,
  MyGridDecoding,
  InitialPoseReady,
  TakeoffTracking,
  HyperGridAcquire,
  HyperGridTracking,
  LandingAcquire,
  LandingTracking,
  Lost,
  Fault,
};

enum class PoseSource : std::uint8_t { None = 0, MyGrid, HyperGrid };
enum class MyGridRequest : std::uint8_t { Blink = 0, Static, Off };

const char *toString(LocalizerState state);
const char *toString(PoseSource source);
const char *toString(MyGridRequest request);

struct Blob {
  cv::Point2f center{};
  cv::Rect bounds{};
  float area = 0.0F;
  float fill_ratio = 0.0F;
  int decoded_id = -1;
  bool used_for_pose = false;
  PoseSource classification = PoseSource::None;
};

struct MatchedPoint {
  std::size_t blob_index = 0;
  cv::Point2f image{};
  cv::Point3f world{};
  int id = -1;
  int grid_x = 0;
  int grid_y = 0;
  int tile_i = 0;
  int tile_j = 0;
  int local_i = -1;
  int local_j = -1;
  float match_error = 0.0F;
};

struct ControllerInput {
  bool attitude_valid = false;
  bool landing_requested = false;
  std::uint32_t ekf_reset_generation = 0;
  double timestamp = 0.0;
  cv::Vec4d quaternion_xyzw{0.0, 0.0, 0.0, 1.0};
  int landing_tile_i = 0;
  int landing_tile_j = 0;
};

struct PoseSolution {
  bool valid = false;
  std::string solver;
  cv::Vec3d tvec_world_to_camera{};
  cv::Matx33d world_to_camera_rotation = cv::Matx33d::eye();
  cv::Vec3d camera_position_world{};
  cv::Vec3d drone_position_world{};
  cv::Vec3d camera_rpy{};
  cv::Vec3d drone_rpy{};
  cv::Vec4d drone_quaternion_xyzw{0.0, 0.0, 0.0, 1.0};
  double camera_to_plane_distance = 0.0;
  double reprojection_rms = std::numeric_limits<double>::infinity();
};

struct GroundTruthEvaluation {
  bool available = false;
  bool pose_evaluated = false;
  std::uint64_t video_frame = 0;
  std::int64_t blender_frame = 0;
  cv::Vec3d position_world_flu{};
  cv::Vec4d quaternion_xyzw{0.0, 0.0, 0.0, 1.0};
  cv::Vec3d position_error_xyz{};
  double position_rmse_frame_m = std::numeric_limits<double>::quiet_NaN();
  double position_rmse_cumulative_m = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t evaluated_pose_count = 0;
};

struct FrameResult {
  std::uint64_t frame_id = 0;
  double timestamp = 0.0;
  double processing_ms = 0.0;
  LocalizerState state = LocalizerState::Starting;
  PoseSource source = PoseSource::None;
  MyGridRequest mygrid_request = MyGridRequest::Blink;
  std::uint32_t initial_pose_generation = 0;
  float hypergrid_acquisition_height_m = 0.0F;
  std::string status = "starting";
  std::string message;
  int tile_i = 0;
  int tile_j = 0;
  std::vector<Blob> blobs;
  std::vector<MatchedPoint> matched;
  PoseSolution pose;
  GroundTruthEvaluation ground_truth;
};

} // namespace flsloc
