#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "application_options.h"
#include "frame_pacer.h"
#include "frame_source.h"

namespace {

constexpr char kWindowName[] = "Marker state evaluator";
constexpr double kDefaultOnThreshold = 0.8;
constexpr double kDefaultOffThreshold = 0.1;
constexpr double kDefaultThresholdStep = 0.01;

std::atomic<bool> keep_running{true};

struct EvaluatorOptions {
  ApplicationOptions capture;
  double on_threshold = kDefaultOnThreshold;
  double off_threshold = kDefaultOffThreshold;
  double threshold_step = kDefaultThresholdStep;
  std::filesystem::path save_directory = ".";
  bool self_test = false;
  bool show_help = false;
};

struct Blob {
  cv::Point center;
  double intensity;
};

void stop(int) { keep_running = false; }

const char *requireValue(const std::string &option, int &index, int argc,
                         char **argv) {
  if (++index >= argc) {
    throw std::invalid_argument(option + " requires a value");
  }
  return argv[index];
}

int parseInteger(const std::string &option, int &index, int argc, char **argv) {
  const std::string text = requireValue(option, index, argc, argv);
  std::size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(option + " requires an integer");
  }
  return value;
}

double parseDouble(const std::string &option, int &index, int argc,
                   char **argv) {
  const std::string text = requireValue(option, index, argc, argv);
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(option + " requires a number");
  }
  return value;
}

EvaluatorOptions parseOptions(int argc, char **argv) {
  EvaluatorOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      options.show_help = true;
    } else if (argument == "--self-test") {
      options.self_test = true;
    } else if (argument == "--video-input") {
      options.capture.video_input_path =
          requireValue(argument, index, argc, argv);
    } else if (argument == "--width") {
      options.capture.cam_width = parseInteger(argument, index, argc, argv);
    } else if (argument == "--height") {
      options.capture.cam_height = parseInteger(argument, index, argc, argv);
    } else if (argument == "--fps") {
      options.capture.frame_rate = parseInteger(argument, index, argc, argv);
    } else if (argument == "--time" || argument == "-t") {
      options.capture.execution_time =
          parseInteger(argument, index, argc, argv);
    } else if (argument == "--contrast") {
      options.capture.contrast = parseDouble(argument, index, argc, argv);
    } else if (argument == "--brightness") {
      options.capture.brightness = parseDouble(argument, index, argc, argv);
    } else if (argument == "--exposure") {
      options.capture.exposure_time = parseInteger(argument, index, argc, argv);
    } else if (argument == "--blob-area-threshold") {
      options.capture.blob_area_threshold =
          parseDouble(argument, index, argc, argv);
    } else if (argument == "--on-threshold") {
      options.on_threshold = parseDouble(argument, index, argc, argv);
    } else if (argument == "--off-threshold") {
      options.off_threshold = parseDouble(argument, index, argc, argv);
    } else if (argument == "--threshold-step") {
      options.threshold_step = parseDouble(argument, index, argc, argv);
    } else if (argument == "--save-dir") {
      options.save_directory = requireValue(argument, index, argc, argv);
    } else if (argument == "--verbose" || argument == "-v") {
      options.capture.print_logs = true;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }

  if (!std::isfinite(options.off_threshold) ||
      !std::isfinite(options.on_threshold) || options.off_threshold < 0.0 ||
      options.off_threshold >= options.on_threshold ||
      options.on_threshold > 1.0) {
    throw std::invalid_argument("thresholds must satisfy 0 <= off < on <= 1");
  }
  if (!std::isfinite(options.threshold_step) || options.threshold_step <= 0.0 ||
      options.threshold_step > 1.0) {
    throw std::invalid_argument("--threshold-step must be in (0, 1]");
  }
  if (!std::isfinite(options.capture.blob_area_threshold) ||
      options.capture.blob_area_threshold < 0.0) {
    throw std::invalid_argument("--blob-area-threshold must be non-negative");
  }
  if (options.capture.frame_rate <= 0 || options.capture.cam_width <= 0 ||
      options.capture.cam_height <= 0 || options.capture.execution_time < 0) {
    throw std::invalid_argument(
        "width, height, and fps must be positive; time cannot be negative");
  }
  return options;
}

void printHelp() {
  std::cout
      << "Usage: marker_state_evaluator [options]\n\n"
      << "Camera options:\n"
      << "  --width N --height N --fps N\n"
      << "  --exposure N --contrast N --brightness N\n"
      << "  --video-input PATH       Video-mode builds only\n"
      << "  --time N                 Stop after N seconds\n\n"
      << "Detection options (normalized grayscale intensity):\n"
      << "  --on-threshold N         Default 0.8\n"
      << "  --off-threshold N        Blob detection floor, default 0.1\n"
      << "  --threshold-step N       Keyboard adjustment, default 0.01\n"
      << "  --blob-area-threshold N  Minimum contour area, default 3\n"
      << "  --save-dir PATH          Saved annotated frames, default .\n\n"
      << "Keys: [ and ] adjust ON, - and = adjust OFF, S save, Q/Esc quit\n"
      << "      F autofocus, A/D manual focus (live camera builds)\n";
}

std::vector<Blob> detectBlobs(const cv::Mat &frame, double off_threshold,
                              double blob_area_threshold) {
  cv::Mat grayscale;
  if (frame.channels() == 1) {
    grayscale = frame.clone();
  } else {
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
  }
  cv::GaussianBlur(grayscale, grayscale, cv::Size(3, 3), 0);

  cv::Mat thresholded;
  cv::threshold(grayscale, thresholded, off_threshold * 255.0, 255,
                cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresholded, contours, cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<Blob> blobs;
  for (const auto &contour : contours) {
    const cv::Moments moments = cv::moments(contour);
    if (moments.m00 <= blob_area_threshold) {
      continue;
    }
    const cv::Point center(static_cast<int>(moments.m10 / moments.m00),
                           static_cast<int>(moments.m01 / moments.m00));
    blobs.push_back({center, grayscale.at<unsigned char>(center) / 255.0});
  }
  return blobs;
}

void putText(cv::Mat &image, const std::string &text, cv::Point origin,
             const cv::Scalar &color, double scale = 0.55) {
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale,
              cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1,
              cv::LINE_AA);
}

std::string thresholdLabel(const char *name, double threshold,
                           const char *keys) {
  std::ostringstream text;
  text << name << " " << keys << ": " << std::fixed << std::setprecision(2)
       << threshold << " (" << static_cast<int>(threshold * 255.0) << ')';
  return text.str();
}

cv::Mat makePreview(const cv::Mat &frame, const std::vector<Blob> &blobs,
                    double on_threshold, double off_threshold) {
  cv::Mat preview;
  if (frame.channels() == 1) {
    cv::cvtColor(frame, preview, cv::COLOR_GRAY2BGR);
  } else {
    preview = frame.clone();
  }

  for (const Blob &blob : blobs) {
    const bool on = blob.intensity >= on_threshold;
    const cv::Scalar color =
        on ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    cv::circle(preview, blob.center, 10, color, 2, cv::LINE_AA);
    std::ostringstream label;
    label << (on ? "ON " : "OFF ") << std::fixed << std::setprecision(2)
          << blob.intensity;
    putText(preview, label.str(), blob.center + cv::Point(12, -8), color);
  }

  const int panel_width = std::min(preview.cols, 390);
  cv::rectangle(preview, {0, 0}, {panel_width, 78}, cv::Scalar(0, 0, 0),
                cv::FILLED);
  putText(preview, thresholdLabel("ON", on_threshold, "[/]"), {8, 22},
          cv::Scalar(0, 255, 0));
  putText(preview, thresholdLabel("OFF", off_threshold, "-/="), {8, 45},
          cv::Scalar(0, 165, 255));
  putText(preview,
          "Blobs: " + std::to_string(blobs.size()) + "   S save   Q/Esc quit",
          {8, 68}, cv::Scalar(255, 255, 255), 0.48);
  return preview;
}

bool adjustThresholds(int key, double step, double &on_threshold,
                      double &off_threshold) {
  if (key == '[') {
    on_threshold =
        std::max(std::nextafter(off_threshold, 1.0), on_threshold - step);
  } else if (key == ']') {
    on_threshold = std::min(1.0, on_threshold + step);
  } else if (key == '-') {
    off_threshold = std::max(0.0, off_threshold - step);
  } else if (key == '=') {
    off_threshold =
        std::min(std::nextafter(on_threshold, 0.0), off_threshold + step);
  } else {
    return false;
  }
  return true;
}

std::filesystem::path framePath(const std::filesystem::path &directory,
                                int frame_id) {
  const std::time_t now = std::time(nullptr);
  std::tm local_time{};
  localtime_r(&now, &local_time);
  std::ostringstream filename;
  filename << "marker_threshold_" << std::put_time(&local_time, "%Y%m%d_%H%M%S")
           << '_' << frame_id << ".png";
  return directory / filename.str();
}

void saveFrame(const cv::Mat &preview, const std::filesystem::path &directory,
               int frame_id) {
  std::filesystem::create_directories(directory);
  const std::filesystem::path path = framePath(directory, frame_id);
  if (!cv::imwrite(path.string(), preview)) {
    throw std::runtime_error("failed to save frame to " + path.string());
  }
  std::cout << "Saved " << path << std::endl;
}

int runSelfTest() {
  cv::Mat frame = cv::Mat::zeros(100, 140, CV_8UC1);
  cv::circle(frame, {35, 50}, 7, cv::Scalar(230), cv::FILLED);
  cv::circle(frame, {105, 50}, 7, cv::Scalar(70), cv::FILLED);
  const std::vector<Blob> blobs = detectBlobs(frame, 0.1, 3.0);
  const int on_count = static_cast<int>(
      std::count_if(blobs.begin(), blobs.end(),
                    [](const Blob &blob) { return blob.intensity >= 0.8; }));
  if (blobs.size() != 2 || on_count != 1) {
    std::cerr << "Self-test failed: expected one ON and one OFF blob\n";
    return 1;
  }
  std::cout << "Marker state evaluator self-test passed\n";
  return 0;
}

int run(const EvaluatorOptions &options) {
  FrameSource source(options.capture);
  FramePacer pacer(source.frameInterval(), options.capture.execution_time,
                   options.capture.print_logs);
  double on_threshold = options.on_threshold;
  double off_threshold = options.off_threshold;
  int frame_id = 0;

  cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
  CapturedFrame frame;
  while (keep_running && source.read(frame)) {
    const std::vector<Blob> blobs = detectBlobs(
        frame.image, off_threshold, options.capture.blob_area_threshold);
    const cv::Mat preview =
        makePreview(frame.image, blobs, on_threshold, off_threshold);
    cv::imshow(kWindowName, preview);

    const int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q' || key == 27) {
      break;
    }
    if (key == 's' || key == 'S') {
      saveFrame(preview, options.save_directory, frame_id);
    } else if (!adjustThresholds(key, options.threshold_step, on_threshold,
                                 off_threshold) &&
               key >= 0) {
      source.handleKey(static_cast<char>(key));
    }

    ++frame_id;
    if (!pacer.completeFrame()) {
      break;
    }
  }
  cv::destroyWindow(kWindowName);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  try {
    const EvaluatorOptions options = parseOptions(argc, argv);
    if (options.show_help) {
      printHelp();
      return 0;
    }
    if (options.self_test) {
      return runSelfTest();
    }
    return run(options);
  } catch (const std::exception &error) {
    std::cerr << "Marker state evaluator error: " << error.what() << '\n';
    return 1;
  }
}
