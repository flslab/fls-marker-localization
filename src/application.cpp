#include "application.h"

#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "application_options.h"
#include "camera_config.h"
#include "frame_outputs.h"
#include "frame_pacer.h"
#include "frame_processor.h"
#include "frame_source.h"
#include "pose_publisher.h"
#include "session_log.h"

namespace {

std::atomic<bool> keep_running{true};

void stop(int) { keep_running = false; }

void printStartup(const ApplicationOptions &options) {
  std::cout << "Running at " << options.frame_rate << " Hz\n"
            << "Decoding marker IDs at " << options.encoder_frame_rate << " Hz"
            << std::endl;
  if (options.static_markers_mode) {
    std::cout << "Static marker blob mode ENABLED" << std::endl;
  }
  if (options.enable_kalman_filter) {
    std::cout << "Kalman filter ENABLED  (process_noise="
              << options.kf_process_noise
              << ", measurement_noise=" << options.kf_measurement_noise << ')'
              << std::endl;
  }
}

} // namespace

int runApplication(int argc, char **argv) {
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);

  try {
    ApplicationOptions options = parseApplicationOptions(argc, argv);
    printStartup(options);

    const std::string log_directory = createSessionDirectory(options);
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    std::vector<cv::Point3f> marker_points;
    if (!readConfigFile(options.config_file, camera_matrix,
                        distortion_coefficients, marker_points)) {
      throw std::runtime_error("Failed to read camera configuration");
    }

    FrameSource source(options);
    std::unique_ptr<FrameProcessor> processor = createFrameProcessor(
        options, camera_matrix, distortion_coefficients, marker_points);
    PosePublisher pose_publisher;
    FrameOutputs outputs(
        options, log_directory, source.windowName(),
        {static_cast<int>(source.width()), static_cast<int>(source.height())});
    SessionLog session_log(options);
    FramePacer pacer(source.frameInterval(), options.execution_time,
                     options.print_logs);

    CapturedFrame frame;
    int frame_id = 0;
    while (keep_running && source.read(frame)) {
      AttitudeSample attitude;
      if (!options.grid_map_file.empty()) {
        pose_publisher.readAttitude(frame.timestamp, options.max_attitude_age,
                                    attitude);
      }
      FrameProcessingResult result = processor->process(
          frame.image, frame.timestamp, attitude);
      pose_publisher.publish(result.pose, frame.timestamp);
      session_log.addFrame(frame_id, frame.timestamp, std::move(result.log));

      const char key = outputs.write(frame, frame_id++);
      if (key == 'q') {
        break;
      }
      source.handleKey(key);
      if (!pacer.completeFrame()) {
        break;
      }
    }

    outputs.finish();
    session_log.save(
        log_directory, *processor,
        {static_cast<int>(source.width()), static_cast<int>(source.height())},
        outputs.videoStartTime());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Application error: " << error.what() << std::endl;
    return -1;
  }
}
