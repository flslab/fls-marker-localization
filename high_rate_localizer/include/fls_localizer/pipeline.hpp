#pragma once

#include "fls_localizer/blink_decoder.hpp"
#include "fls_localizer/blob_detector.hpp"
#include "fls_localizer/config.hpp"
#include "fls_localizer/grid_map.hpp"
#include "fls_localizer/hypergrid.hpp"
#include "fls_localizer/pose_solver.hpp"
#include "fls_localizer/types.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace flsloc {

class LocalizationPipeline {
public:
  LocalizationPipeline(ApplicationConfig config, GridMap map);

  FrameResult process(std::uint64_t frame_id, double timestamp,
                      const cv::Mat &gray, const ControllerInput &controller);

  const GridMap &gridMap() const { return map_; }
  const ApplicationConfig &config() const { return config_; }

private:
  std::vector<MatchedPoint>
  initialMatches(const DecodedRing &ring, const SignatureMatch &signature,
                 const std::vector<Blob> &blobs) const;
  std::vector<MatchedPoint> matchKnownTile(const MyGridTile &tile,
                                           const std::vector<Blob> &blobs,
                                           const cv::Vec3d &camera_position,
                                           const cv::Matx33d &world_to_camera,
                                           double projection_gate_px) const;
  bool acceptable(const PoseSolution &pose) const;
  cv::Vec3d predictedCameraPosition(double timestamp) const;
  void usePose(FrameResult &result, PoseSource source,
               std::vector<MatchedPoint> matches, PoseSolution pose);
  void setIdleStatus(FrameResult &result) const;

  ApplicationConfig config_;
  GridMap map_;
  BlobDetector detector_;
  BlinkDecoder blink_decoder_;
  PoseSolver pose_solver_;
  HyperGridMatcher hypergrid_matcher_;
  LocalizerState state_ = LocalizerState::MyGridDecoding;
  MyGridRequest mygrid_request_ = MyGridRequest::Blink;
  const MyGridTile *start_tile_ = nullptr;
  PoseSolution last_pose_;
  cv::Vec3d camera_velocity_world_{};
  double last_pose_timestamp_ = -1.0;
  std::uint32_t initial_pose_generation_ = 0;
  int hypergrid_confirmations_ = 0;
  int invalid_frames_ = 0;
  float hypergrid_acquisition_height_m_ = 0.0F;
  std::uint32_t session_generation_ = 0;
};

} // namespace flsloc
