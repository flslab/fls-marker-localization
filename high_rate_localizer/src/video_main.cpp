#include "fls_localizer/config.hpp"
#include "fls_localizer/debug_output.hpp"
#include "fls_localizer/grid_map.hpp"
#include "fls_localizer/pipeline.hpp"
#include "fls_localizer/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
  std::filesystem::path config;
  std::filesystem::path video;
  std::filesystem::path grid;
  std::filesystem::path trajectory;
  std::filesystem::path output;
  flsloc::OrientationErrorConfig orientation_error;
};

double parseNumber(const std::string &value, const std::string &option) {
  std::size_t parsed = 0;
  double result = 0.0;
  try {
    result = std::stod(value, &parsed);
  } catch (const std::exception &) {
    throw std::runtime_error(option + " requires a numeric value");
  }
  if (parsed != value.size() || !std::isfinite(result)) {
    throw std::runtime_error(option + " requires a finite numeric value");
  }
  return result;
}

cv::Vec3d parseRpyDegrees(const std::string &value, const std::string &option) {
  std::stringstream input(value);
  std::vector<double> values;
  std::string component;
  while (std::getline(input, component, ',')) {
    values.push_back(parseNumber(component, option));
  }
  if (values.size() == 1) {
    values.resize(3, values.front());
  }
  if (values.size() != 3) {
    throw std::runtime_error(option + " requires DEG or ROLL,PITCH,YAW");
  }
  constexpr double degrees_to_radians = std::numbers::pi / 180.0;
  return {values[0] * degrees_to_radians, values[1] * degrees_to_radians,
          values[2] * degrees_to_radians};
}

std::uint64_t parseSeed(const std::string &value) {
  if (value.empty() || value.front() == '-') {
    throw std::runtime_error(
        "--orientation-noise-seed requires an unsigned integer");
  }
  std::size_t parsed = 0;
  unsigned long long seed = 0;
  try {
    seed = std::stoull(value, &parsed);
  } catch (const std::exception &) {
    throw std::runtime_error(
        "--orientation-noise-seed requires an unsigned integer");
  }
  if (parsed != value.size()) {
    throw std::runtime_error(
        "--orientation-noise-seed requires an unsigned integer");
  }
  return static_cast<std::uint64_t>(seed);
}

Arguments parse(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value after " + option);
    }
    if (option == "--config") {
      result.config = argv[++index];
    } else if (option == "--video") {
      result.video = argv[++index];
    } else if (option == "--grid") {
      result.grid = argv[++index];
    } else if (option == "--trajectory") {
      result.trajectory = argv[++index];
    } else if (option == "--output-dir") {
      result.output = argv[++index];
    } else if (option == "--orientation-bias-deg") {
      result.orientation_error.bias_rpy_rad =
          parseRpyDegrees(argv[++index], option);
    } else if (option == "--orientation-noise-stddev-deg") {
      result.orientation_error.noise_stddev_rpy_rad =
          parseRpyDegrees(argv[++index], option);
      for (double standard_deviation :
           result.orientation_error.noise_stddev_rpy_rad.val) {
        if (standard_deviation < 0.0) {
          throw std::runtime_error(option + " cannot be negative");
        }
      }
    } else if (option == "--orientation-noise-seed") {
      result.orientation_error.seed = parseSeed(argv[++index]);
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.config.empty() || result.video.empty() ||
      result.trajectory.empty()) {
    throw std::runtime_error(
        "usage: fls_localizer_video --config FILE --video FILE "
        "--trajectory FILE [--grid FILE] [--output-dir DIR] "
        "[--orientation-bias-deg ROLL,PITCH,YAW] "
        "[--orientation-noise-stddev-deg ROLL,PITCH,YAW] "
        "[--orientation-noise-seed N]");
  }
  return result;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

} // namespace

int main(int argc, char **argv) try {
  const Arguments arguments = parse(argc, argv);
  flsloc::ApplicationConfig config =
      flsloc::loadApplicationConfig(arguments.config);
  if (!arguments.grid.empty()) {
    config.grid_file = arguments.grid;
  }
  if (!arguments.output.empty()) {
    config.output.directory = arguments.output;
  }
  flsloc::GridMap map = flsloc::GridMap::load(config.grid_file);
  const flsloc::GroundTruthTrajectory trajectory =
      flsloc::GroundTruthTrajectory::load(arguments.trajectory);
  flsloc::OrientationErrorModel orientation_error(arguments.orientation_error);
  flsloc::LocalizationPipeline pipeline(config, map);
  flsloc::DebugOutput output(config, map, arguments.video.string(),
                             arguments.trajectory.string(),
                             arguments.orientation_error);

  cv::VideoCapture capture(arguments.video.string());
  if (!capture.isOpened()) {
    throw std::runtime_error("unable to open video " +
                             arguments.video.string());
  }
  const double frame_rate = capture.get(cv::CAP_PROP_FPS);
  if (frame_rate <= 0.0 ||
      std::abs(frame_rate - trajectory.frameRate()) > 1e-3) {
    throw std::runtime_error("video and trajectory frame rates do not match");
  }
  const auto video_frame_count = static_cast<std::uint64_t>(
      std::llround(capture.get(cv::CAP_PROP_FRAME_COUNT)));
  if (video_frame_count > 0 && trajectory.size() != video_frame_count) {
    throw std::runtime_error("video and trajectory frame counts do not match");
  }
  cv::Mat bgr;
  cv::Mat gray;
  flsloc::ControllerInput controller;
  controller.landing_tile_i = config.default_landing_tile_i;
  controller.landing_tile_j = config.default_landing_tile_j;
  std::uint64_t frame_id = 0;
  bool initialized = false;
  bool hypergrid_locked = false;
  bool landing_locked = false;
  std::vector<double> timings;
  flsloc::TrajectoryEvaluator evaluator;

  while (capture.read(bgr)) {
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    const double timestamp = frame_rate > 0.0
                                 ? static_cast<double>(frame_id) / frame_rate
                                 : capture.get(cv::CAP_PROP_POS_MSEC) * 1e-3;
    controller.timestamp = timestamp;
    const flsloc::TrajectorySample &truth = trajectory.sample(frame_id);
    controller.quaternion_xyzw = orientation_error.apply(truth.quaternion_xyzw);
    controller.landing_requested =
        timestamp >= config.video_test_landing_time_s;
    flsloc::FrameResult result =
        pipeline.process(frame_id, timestamp, gray, controller);
    evaluator.evaluate(truth, result);
    result.ground_truth.simulated_ekf_quaternion_xyzw =
        controller.quaternion_xyzw;
    timings.push_back(result.processing_ms);
    initialized = initialized || result.initial_pose_generation > 0;
    hypergrid_locked =
        hypergrid_locked ||
        result.state == flsloc::LocalizerState::HyperGridTracking;
    landing_locked = landing_locked ||
                     result.state == flsloc::LocalizerState::LandingTracking;
    if (result.initial_pose_generation > controller.ekf_reset_generation &&
        result.trackingPose().accepted) {
      controller.attitude_valid = true;
      controller.ekf_reset_generation = result.initial_pose_generation;
    }
    cv::Mat annotated;
    if (output.wantsVideoFrame(timestamp)) {
      annotated = output.annotate(bgr, result);
    }
    output.submit(std::move(result), std::move(annotated));
    ++frame_id;
  }
  output.finish();
  if (frame_id != trajectory.size()) {
    throw std::runtime_error("trajectory has more frames than the video");
  }

  std::cout << "frames=" << frame_id << " p50_ms=" << percentile(timings, 0.50)
            << " p99_ms=" << percentile(timings, 0.99)
            << " initialized=" << initialized
            << " hypergrid_locked=" << hypergrid_locked
            << " landing_locked=" << landing_locked
            << " position_rmse_m=" << evaluator.cumulativePositionRmse()
            << " evaluated_poses=" << evaluator.evaluatedPoseCount() << '\n'
            << "log=" << output.logPath() << '\n'
            << "annotated_video=" << output.videoPath() << std::endl;
  return initialized && hypergrid_locked && landing_locked ? 0 : 2;
} catch (const std::exception &error) {
  std::cerr << "fls_localizer_video: " << error.what() << std::endl;
  return 1;
}
