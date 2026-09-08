#include "fls_localizer/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>

namespace flsloc {

const char *toString(LocalizerState state) {
  switch (state) {
  case LocalizerState::Starting:
    return "starting";
  case LocalizerState::MyGridDecoding:
    return "mygrid_decoding";
  case LocalizerState::InitialPoseReady:
    return "initial_pose_ready";
  case LocalizerState::TakeoffTracking:
    return "takeoff_tracking";
  case LocalizerState::HyperGridAcquire:
    return "hypergrid_acquire";
  case LocalizerState::HyperGridTracking:
    return "hypergrid_tracking";
  case LocalizerState::LandingAcquire:
    return "landing_acquire";
  case LocalizerState::LandingTracking:
    return "landing_tracking";
  case LocalizerState::Lost:
    return "lost";
  case LocalizerState::Fault:
    return "fault";
  }
  return "fault";
}

const char *toString(PoseSource source) {
  switch (source) {
  case PoseSource::None:
    return "none";
  case PoseSource::MyGrid:
    return "mygrid";
  case PoseSource::HyperGrid:
    return "hypergrid";
  }
  return "none";
}

const char *toString(MyGridRequest request) {
  switch (request) {
  case MyGridRequest::Blink:
    return "blink";
  case MyGridRequest::Static:
    return "static";
  case MyGridRequest::Off:
    return "off";
  }
  return "blink";
}

LocalizationPipeline::LocalizationPipeline(ApplicationConfig config,
                                           GridMap map)
    : config_(std::move(config)), map_(std::move(map)),
      detector_(config_.detector),
      blink_decoder_(map_, config_.detector.intensity_threshold,
                     config_.tracking.projection_gate_px),
      pose_solver_(config_), hypergrid_matcher_(map_, config_) {
  const double fx = config_.calibration.camera_matrix.at<double>(0, 0);
  const double fy = config_.calibration.camera_matrix.at<double>(1, 1);
  const double height =
      2.0 * map_.hypergridSpacing() *
      std::max(fx / config_.camera.width, fy / config_.camera.height) * 1.10;
  hypergrid_acquisition_height_m_ = static_cast<float>(height);
  const auto ticks = static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  session_generation_ = static_cast<std::uint32_t>(ticks) ^
                        static_cast<std::uint32_t>(ticks >> 32);
  if (session_generation_ == 0) {
    session_generation_ = 1;
  }
}

std::vector<MatchedPoint>
LocalizationPipeline::initialMatches(const DecodedRing &ring,
                                     const SignatureMatch &signature,
                                     const std::vector<Blob> &blobs) const {
  std::vector<MatchedPoint> matches;
  matches.reserve(4);
  for (int observed = 0; observed < 4; ++observed) {
    const int map_index = (observed + signature.rotation) % 4;
    const MyGridMarker &marker = signature.tile->markers[map_index];
    MatchedPoint match;
    match.image = ring.image_points[observed];
    match.world = marker.world;
    match.id = marker.id;
    match.tile_i = signature.tile->i;
    match.tile_j = signature.tile->j;
    match.local_i = marker.local_i;
    match.local_j = marker.local_j;
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t blob = 0; blob < blobs.size(); ++blob) {
      const double distance = cv::norm(blobs[blob].center - match.image);
      if (distance < nearest) {
        nearest = distance;
        match.blob_index = blob;
      }
    }
    matches.push_back(match);
  }
  return matches;
}

std::vector<MatchedPoint> LocalizationPipeline::matchKnownTile(
    const MyGridTile &tile, const std::vector<Blob> &blobs,
    const cv::Vec3d &camera_position, const cv::Matx33d &world_to_camera,
    double projection_gate_px) const {
  if (blobs.empty()) {
    return {};
  }
  std::vector<cv::Point3f> world;
  world.reserve(4);
  for (const MyGridMarker &marker : tile.markers) {
    world.push_back(marker.world);
  }
  const cv::Vec3d translation = -(world_to_camera * camera_position);
  cv::Vec3d rvec;
  cv::Rodrigues(world_to_camera, rvec);
  std::vector<cv::Point2f> projected;
  cv::projectPoints(world, rvec, translation, config_.calibration.camera_matrix,
                    config_.calibration.distortion, projected);

  std::vector<MatchedPoint> matches;
  std::vector<bool> used(blobs.size(), false);
  for (int marker_index = 0; marker_index < 4; ++marker_index) {
    double best_distance = projection_gate_px;
    int best_blob = -1;
    for (std::size_t blob_index = 0; blob_index < blobs.size(); ++blob_index) {
      if (used[blob_index]) {
        continue;
      }
      const double distance =
          cv::norm(blobs[blob_index].center - projected[marker_index]);
      if (distance < best_distance) {
        best_distance = distance;
        best_blob = static_cast<int>(blob_index);
      }
    }
    if (best_blob < 0) {
      continue;
    }
    used[best_blob] = true;
    const MyGridMarker &marker = tile.markers[marker_index];
    MatchedPoint match;
    match.blob_index = static_cast<std::size_t>(best_blob);
    match.image = blobs[best_blob].center;
    match.world = marker.world;
    match.id = marker.id;
    match.tile_i = tile.i;
    match.tile_j = tile.j;
    match.local_i = marker.local_i;
    match.local_j = marker.local_j;
    match.match_error = static_cast<float>(best_distance);
    matches.push_back(match);
  }
  return matches;
}

bool LocalizationPipeline::acceptable(const PoseSolution &pose) const {
  return pose.valid && pose.reprojection_rms <=
                           config_.tracking.maximum_reprojection_error_px;
}

cv::Vec3d
LocalizationPipeline::predictedCameraPosition(double timestamp) const {
  if (!last_pose_.valid || last_pose_timestamp_ < 0.0) {
    return last_pose_.camera_position_world;
  }
  const double elapsed =
      std::clamp(timestamp - last_pose_timestamp_, 0.0, 0.30);
  return last_pose_.camera_position_world + elapsed * camera_velocity_world_;
}

void LocalizationPipeline::usePose(FrameResult &result, PoseSource source,
                                   std::vector<MatchedPoint> matches,
                                   PoseSolution pose) {
  result.source = source;
  result.pose = std::move(pose);
  result.matched = std::move(matches);
  result.status = "success";
  result.message = source == PoseSource::HyperGrid
                       ? "pose solved from static HyperGrid lattice"
                       : "pose solved from MyGrid landing markers";
  for (const MatchedPoint &match : result.matched) {
    if (match.blob_index < result.blobs.size()) {
      Blob &blob = result.blobs[match.blob_index];
      blob.used_for_pose = true;
      blob.classification = source;
      blob.decoded_id = match.id;
    }
  }
  if (last_pose_.valid && last_pose_timestamp_ >= 0.0) {
    const double elapsed = result.timestamp - last_pose_timestamp_;
    if (elapsed > 1e-5 && elapsed < 0.5) {
      cv::Vec3d measured_velocity = (result.pose.camera_position_world -
                                     last_pose_.camera_position_world) /
                                    elapsed;
      const double speed = cv::norm(measured_velocity);
      if (speed > 1.5) {
        measured_velocity *= 1.5 / speed;
      }
      camera_velocity_world_ =
          0.65 * camera_velocity_world_ + 0.35 * measured_velocity;
    }
  }
  last_pose_ = result.pose;
  last_pose_timestamp_ = result.timestamp;
  invalid_frames_ = 0;
}

void LocalizationPipeline::setIdleStatus(FrameResult &result) const {
  result.status =
      result.blobs.empty() ? "no_detections" : "insufficient_geometry";
  result.message = result.blobs.empty()
                       ? "no marker blobs detected"
                       : "detected blobs did not form a valid pose";
}

FrameResult LocalizationPipeline::process(std::uint64_t frame_id,
                                          double timestamp, const cv::Mat &gray,
                                          const ControllerInput &controller) {
  const auto started = std::chrono::steady_clock::now();
  FrameResult result;
  result.frame_id = frame_id;
  result.timestamp = timestamp;
  result.state = state_;
  result.mygrid_request = mygrid_request_;
  result.initial_pose_generation = initial_pose_generation_;
  result.hypergrid_acquisition_height_m = hypergrid_acquisition_height_m_;
  result.blobs = detector_.detect(gray);

  if (state_ == LocalizerState::MyGridDecoding) {
    const auto decoded = blink_decoder_.update(timestamp, gray, result.blobs);
    if (decoded) {
      const auto signature = map_.matchSignature(decoded->ids);
      if (signature) {
        auto matches = initialMatches(*decoded, *signature, result.blobs);
        PoseSolution pose = pose_solver_.solveInitial(
            matches, config_.tracking.initial_distance_m);
        if (acceptable(pose)) {
          start_tile_ = signature->tile;
          initial_pose_generation_ = session_generation_;
          state_ = LocalizerState::InitialPoseReady;
          mygrid_request_ = MyGridRequest::Static;
          result.tile_i = start_tile_->i;
          result.tile_j = start_tile_->j;
          usePose(result, PoseSource::MyGrid, std::move(matches),
                  std::move(pose));
        } else {
          result.status = "pnp_failed";
          result.message = "decoded MyGrid ring but IPPE rejected the pose";
        }
      } else {
        result.status = "signature_not_unique";
        result.message = "decoded ring was not unique in the MyGrid map";
      }
    } else {
      result.status = "decoding";
      result.message = "collecting MyGrid blink signature";
    }
  } else if (state_ == LocalizerState::InitialPoseReady) {
    if (controller.attitude_valid &&
        controller.ekf_reset_generation == initial_pose_generation_) {
      state_ = LocalizerState::TakeoffTracking;
    }
    result.pose = last_pose_;
    result.source = PoseSource::MyGrid;
    result.tile_i = start_tile_ ? start_tile_->i : 0;
    result.tile_j = start_tile_ ? start_tile_->j : 0;
    result.status = "waiting_for_ekf_reset";
    result.message = "initial yaw and position are ready for the controller";
  }

  if (state_ != LocalizerState::MyGridDecoding &&
      state_ != LocalizerState::InitialPoseReady) {
    const bool attitude_fresh = controller.attitude_valid &&
                                (controller.timestamp <= 0.0 ||
                                 std::abs(timestamp - controller.timestamp) <=
                                     config_.tracking.maximum_attitude_age_s);
    if (!attitude_fresh || !last_pose_.valid) {
      ++invalid_frames_;
      result.status = "attitude_unavailable";
      result.message = "shared EKF quaternion is invalid or stale";
    } else {
      const cv::Matx33d world_to_camera =
          pose_solver_.worldToCameraFromDrone(controller.quaternion_xyzw);
      const cv::Vec3d predicted_camera = predictedCameraPosition(timestamp);
      const double predicted_distance = predicted_camera[2] - map_.origin().z;
      const MyGridTile *landing_tile =
          map_.findTile(controller.landing_tile_i, controller.landing_tile_j);
      if (!landing_tile) {
        landing_tile = map_.findTile(config_.default_landing_tile_i,
                                     config_.default_landing_tile_j);
      }

      if (controller.landing_requested &&
          state_ != LocalizerState::LandingTracking) {
        state_ = LocalizerState::LandingAcquire;
        mygrid_request_ = MyGridRequest::Static;
      }

      bool pose_used = false;
      if (state_ == LocalizerState::LandingAcquire ||
          state_ == LocalizerState::LandingTracking) {
        if (landing_tile) {
          auto matches = matchKnownTile(
              *landing_tile, result.blobs, predicted_camera, world_to_camera,
              config_.tracking.projection_gate_px * 4.0);
          if (matches.size() >= 4) {
            PoseSolution pose = pose_solver_.solveWithAttitude(
                matches, controller.quaternion_xyzw, predicted_distance);
            if (acceptable(pose)) {
              state_ = LocalizerState::LandingTracking;
              result.tile_i = landing_tile->i;
              result.tile_j = landing_tile->j;
              usePose(result, PoseSource::MyGrid, std::move(matches),
                      std::move(pose));
              pose_used = true;
            }
          }
        }
      }

      if (!pose_used) {
        auto hyper_matches = hypergrid_matcher_.match(
            result.blobs, predicted_camera, world_to_camera);
        if (hyper_matches.size() >= 4) {
          PoseSolution pose = pose_solver_.solveWithAttitude(
              hyper_matches, controller.quaternion_xyzw, predicted_distance);
          if (acceptable(pose)) {
            if (state_ == LocalizerState::Lost &&
                !controller.landing_requested) {
              state_ = LocalizerState::HyperGridTracking;
            }
            if (state_ == LocalizerState::TakeoffTracking ||
                state_ == LocalizerState::HyperGridAcquire) {
              ++hypergrid_confirmations_;
              state_ = hypergrid_confirmations_ >=
                               config_.tracking.hypergrid_confirmation_frames
                           ? LocalizerState::HyperGridTracking
                           : LocalizerState::HyperGridAcquire;
            }
            if (state_ == LocalizerState::HyperGridTracking) {
              mygrid_request_ = MyGridRequest::Off;
            }
            usePose(result, PoseSource::HyperGrid, std::move(hyper_matches),
                    std::move(pose));
            pose_used = true;
          }
        }
      }

      if (!pose_used && state_ == LocalizerState::TakeoffTracking &&
          start_tile_) {
        auto matches = matchKnownTile(*start_tile_, result.blobs,
                                      predicted_camera, world_to_camera,
                                      config_.tracking.projection_gate_px);
        if (matches.size() >= 2) {
          PoseSolution pose = pose_solver_.solveWithAttitude(
              matches, controller.quaternion_xyzw, predicted_distance);
          if (acceptable(pose)) {
            usePose(result, PoseSource::MyGrid, std::move(matches),
                    std::move(pose));
            pose_used = true;
          }
        }
      }

      if (!pose_used) {
        ++invalid_frames_;
        setIdleStatus(result);
        if (state_ == LocalizerState::HyperGridAcquire) {
          state_ = LocalizerState::TakeoffTracking;
          hypergrid_confirmations_ = 0;
        }
      }
    }
  }

  if (invalid_frames_ > config_.tracking.lost_after_frames &&
      state_ != LocalizerState::MyGridDecoding &&
      state_ != LocalizerState::InitialPoseReady) {
    state_ = LocalizerState::Lost;
    result.status = "lost";
    result.message = "no geometrically valid pose within the loss window";
  }
  result.state = state_;
  result.mygrid_request = mygrid_request_;
  result.initial_pose_generation = initial_pose_generation_;
  result.processing_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  return result;
}

} // namespace flsloc
