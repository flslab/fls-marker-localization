#include "fls_localizer/blink_decoder.hpp"
#include "fls_localizer/config.hpp"
#include "fls_localizer/grid_map.hpp"
#include "fls_localizer/hypergrid.hpp"
#include "fls_localizer/pipeline.hpp"
#include "fls_localizer/pose_solver.hpp"
#include "fls_localizer/trajectory.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

flsloc::ApplicationConfig testConfig() {
  flsloc::ApplicationConfig config;
  config.calibration.camera_matrix = (cv::Mat_<double>(3, 3) << 478.0, 0.0,
                                      320.0, 0.0, 478.0, 200.0, 0.0, 0.0, 1.0);
  config.calibration.distortion = cv::Mat::zeros(5, 1, CV_64F);
  config.tracking.maximum_pose_points = 16;
  config.tracking.lattice_tolerance_cells = 0.45;
  return config;
}

void testGrid(const flsloc::GridMap &map) {
  require(std::abs(map.hypergridSpacing() - 0.155) < 1e-9,
          "HyperGrid spacing was not loaded");
  const auto match = map.matchSignature({5, 0, 0, 0});
  require(match.has_value(), "cyclic MyGrid ring did not match");
  require(match->tile->i == 0 && match->tile->j == 0,
          "MyGrid ring matched the wrong tile");
  const cv::Point3f point = map.hypergridPoint(0, 0);
  require(std::abs(point.x - 0.0775F) < 1e-6F &&
              std::abs(point.y - 0.0775F) < 1e-6F,
          "HyperGrid coordinate convention changed");
}

void testBlink(const flsloc::GridMap &map) {
  flsloc::BlinkDecoder decoder(map, 160, 35.0);
  const std::array<cv::Point2f, 4> centers{
      cv::Point2f(200, 100), cv::Point2f(440, 100), cv::Point2f(440, 340),
      cv::Point2f(200, 340)};
  const std::array<int, 4> ids{0, 0, 0, 5};
  const int packet_bits =
      map.payloadBits() + static_cast<int>(map.delimiterPattern().size());
  std::optional<flsloc::DecodedRing> decoded;
  for (int frame = 0; frame < 120; ++frame) {
    const double time = frame / 120.0;
    const int bit =
        static_cast<int>(time / map.bitDurationSeconds()) % packet_bits;
    cv::Mat gray = cv::Mat::zeros(400, 640, CV_8UC1);
    std::vector<flsloc::Blob> blobs;
    for (int slot = 0; slot < 4; ++slot) {
      bool on = false;
      if (bit < map.payloadBits()) {
        on = ((ids[slot] >> (map.payloadBits() - bit - 1)) & 1) != 0;
      } else {
        on = map.delimiterPattern()[bit - map.payloadBits()] == '1';
      }
      if (on) {
        cv::circle(gray, centers[slot], 8, cv::Scalar(255), -1);
        flsloc::Blob blob;
        blob.center = centers[slot];
        blob.area = 200;
        blob.bounds = {static_cast<int>(centers[slot].x - 8),
                       static_cast<int>(centers[slot].y - 8), 17, 17};
        blob.fill_ratio = 0.7F;
        blobs.push_back(blob);
      }
    }
    const auto current = decoder.update(time, gray, blobs);
    if (current) {
      decoded = current;
    }
  }
  require(decoded.has_value(), "blink decoder did not find a packet");
  require(decoded->ids == ids, "blink decoder returned the wrong IDs");
}

void testPose(const flsloc::GridMap &map) {
  flsloc::ApplicationConfig config = testConfig();
  flsloc::PoseSolver solver(config);
  const flsloc::MyGridTile *tile = map.findTile(0, 0);
  require(tile != nullptr, "test tile missing");
  const cv::Vec3d expected_camera(0.0, 0.0, 0.20);
  const cv::Matx33d rotation =
      solver.worldToCameraFromDrone({0.0, 0.0, 0.0, 1.0});
  const cv::Vec3d translation = -(rotation * expected_camera);
  cv::Vec3d rvec;
  cv::Rodrigues(rotation, rvec);
  std::vector<cv::Point3f> world;
  for (const flsloc::MyGridMarker &marker : tile->markers) {
    world.push_back(marker.world);
  }
  std::vector<cv::Point2f> image;
  cv::projectPoints(world, rvec, translation, config.calibration.camera_matrix,
                    config.calibration.distortion, image);
  std::vector<flsloc::MatchedPoint> matches;
  for (int index = 0; index < 4; ++index) {
    flsloc::MatchedPoint match;
    match.image = image[index];
    match.world = world[index];
    matches.push_back(match);
  }
  const flsloc::PoseSolution pose =
      solver.solveWithAttitude(matches, {0.0, 0.0, 0.0, 1.0}, 0.20);
  require(pose.valid, "IPPE/shared-attitude pose failed");
  require(cv::norm(pose.camera_position_world - expected_camera) < 1e-6,
          "FLU position transform is incorrect");
}

void testTakeoffAttitudeAcquisition(const flsloc::GridMap &map) {
  flsloc::ApplicationConfig config = testConfig();
  config.calibration.camera_matrix =
      (cv::Mat_<double>(3, 3) << 478.11017984, 0.0, 322.59805209, 0.0,
       478.29786406, 195.78709198, 0.0, 0.0, 1.0);
  config.calibration.distortion =
      (cv::Mat_<double>(5, 1) << 0.159361045, 0.00175631861, -0.000966795628,
       0.001165244, -1.18066737);
  config.tracking.initial_distance_m = 0.045;
  config.tracking.maximum_reprojection_error_px = 5.0;
  config.tracking.projection_gate_px = 35.0;

  const flsloc::MyGridTile *tile = map.findTile(0, 0);
  require(tile != nullptr, "takeoff test tile missing");
  const std::array<cv::Point2f, 4> centers{
      cv::Point2f(432.880F, 336.835F), cv::Point2f(157.069F, 323.528F),
      cv::Point2f(166.407F, 47.305F), cv::Point2f(447.958F, 60.872F)};
  const int packet_bits =
      map.payloadBits() + static_cast<int>(map.delimiterPattern().size());
  flsloc::LocalizationPipeline pipeline(config, map);
  flsloc::FrameResult initial;
  std::uint64_t frame = 0;
  double timestamp = 0.0;
  for (; frame < 300; ++frame) {
    timestamp = frame / 120.0;
    const int bit =
        static_cast<int>(timestamp / map.bitDurationSeconds()) % packet_bits;
    cv::Mat gray = cv::Mat::zeros(400, 640, CV_8UC1);
    for (int marker = 0; marker < 4; ++marker) {
      const int id = tile->markers[marker].id;
      const bool on =
          bit < map.payloadBits()
              ? ((id >> (map.payloadBits() - bit - 1)) & 1) != 0
              : map.delimiterPattern()[bit - map.payloadBits()] == '1';
      if (on) {
        cv::circle(gray, centers[marker], 8, cv::Scalar(255), -1);
      }
    }
    initial = pipeline.process(frame, timestamp, gray, {});
    if (initial.state == flsloc::LocalizerState::InitialPoseReady) {
      break;
    }
  }
  require(initial.state == flsloc::LocalizerState::InitialPoseReady,
          "takeoff test did not decode the initial pose");

  cv::Mat static_markers = cv::Mat::zeros(400, 640, CV_8UC1);
  for (const cv::Point2f &center : centers) {
    cv::circle(static_markers, center, 8, cv::Scalar(255), -1);
  }
  const double yaw = -1.522218;
  flsloc::ControllerInput controller;
  controller.attitude_valid = true;
  controller.ekf_reset_generation = initial.initial_pose_generation;
  controller.timestamp = timestamp + 1.0 / 120.0;
  controller.quaternion_xyzw = {0.0, 0.0, std::sin(yaw / 2.0),
                                std::cos(yaw / 2.0)};
  const flsloc::FrameResult acquired = pipeline.process(
      ++frame, controller.timestamp, static_markers, controller);
  require(acquired.state == flsloc::LocalizerState::TakeoffTracking,
          "takeoff did not enter shared-attitude tracking");
  require(acquired.pose.valid && acquired.matched.size() == 4,
          "wide takeoff acquisition did not recover the known tile");
  require(acquired.pose.reprojection_rms < 2.0,
          "shared-attitude takeoff pose was not geometrically valid");

  controller.timestamp += 1.0 / 120.0;
  const flsloc::FrameResult tracked = pipeline.process(
      ++frame, controller.timestamp, static_markers, controller);
  require(tracked.pose.valid && tracked.matched.size() == 4,
          "normal projection gate did not retain the acquired takeoff pose");
}

void testHyperGrid(const flsloc::GridMap &map) {
  flsloc::ApplicationConfig config = testConfig();
  flsloc::PoseSolver solver(config);
  flsloc::HyperGridMatcher matcher(map, config);
  const cv::Vec3d camera(0.02, -0.01, 0.55);
  const cv::Matx33d rotation =
      solver.worldToCameraFromDrone({0.0, 0.0, 0.0, 1.0});
  const cv::Vec3d translation = -(rotation * camera);
  cv::Vec3d rvec;
  cv::Rodrigues(rotation, rvec);
  std::vector<cv::Point3f> world;
  for (int x : {-1, 0, 1}) {
    for (int y : {-1, 0, 1}) {
      world.push_back(map.hypergridPoint(x, y));
    }
  }
  std::vector<cv::Point2f> image;
  cv::projectPoints(world, rvec, translation, config.calibration.camera_matrix,
                    config.calibration.distortion, image);
  std::vector<flsloc::Blob> blobs;
  for (const cv::Point2f &point : image) {
    flsloc::Blob blob;
    blob.center = point;
    blob.area = 50.0F;
    blobs.push_back(blob);
  }
  const auto matches = matcher.match(blobs, camera, rotation);
  require(matches.size() == world.size(),
          "HyperGrid coordinates were not assigned immediately");
}

void testTrajectory() {
  const flsloc::GroundTruthTrajectory trajectory =
      flsloc::GroundTruthTrajectory::load(FLS_TEST_TRAJECTORY_PATH);
  require(trajectory.size() == 2, "trajectory frame count was not loaded");
  require(std::abs(trajectory.frameRate() - 120.0) < 1e-9,
          "trajectory frame rate was not loaded");
  require(cv::norm(trajectory.sample(1).position_world_flu -
                   cv::Vec3d(1.1, 2.2, 3.3)) < 1e-12,
          "trajectory position was not loaded");

  flsloc::TrajectoryEvaluator evaluator;
  flsloc::FrameResult first;
  first.pose.valid = true;
  first.pose.drone_position_world = {1.1, 1.8, 3.3};
  evaluator.evaluate(trajectory.sample(0), first);
  const double first_squared_error = 0.01 + 0.04 + 0.09;
  require(std::abs(first.ground_truth.position_rmse_frame_m -
                   std::sqrt(first_squared_error)) < 1e-12,
          "per-frame position RMSE is incorrect");

  flsloc::FrameResult second;
  second.pose.valid = true;
  second.pose.drone_position_world = trajectory.sample(1).position_world_flu;
  evaluator.evaluate(trajectory.sample(1), second);
  require(std::abs(second.ground_truth.position_rmse_cumulative_m -
                   std::sqrt(first_squared_error / 2.0)) < 1e-12,
          "cumulative position RMSE is incorrect");
}

} // namespace

int main() try {
  const flsloc::GridMap map = flsloc::GridMap::load(FLS_TEST_GRID_PATH);
  testGrid(map);
  testBlink(map);
  testPose(map);
  testTakeoffAttitudeAcquisition(map);
  testHyperGrid(map);
  testTrajectory();
  std::cout << "all high-rate localizer tests passed" << std::endl;
  return 0;
} catch (const std::exception &error) {
  std::cerr << "test failure: " << error.what() << std::endl;
  return 1;
}
