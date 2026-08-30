#include "frame_source.h"

#include <ctime>
#include <iostream>
#include <stdexcept>

#include "application_options.h"

#ifdef VIDEO_INPUT_MODE

#include <opencv2/videoio.hpp>

class FrameSource::Impl {
public:
  explicit Impl(const ApplicationOptions &options)
      : capture(options.video_input_path) {
    if (options.video_input_path.empty()) {
      throw std::runtime_error(
          "Built in VIDEO_INPUT_MODE but no --video-input <path> was provided");
    }
    if (!capture.isOpened()) {
      throw std::runtime_error("Could not open video file: " +
                               options.video_input_path);
    }

    double native_fps = capture.get(cv::CAP_PROP_FPS);
    if (native_fps <= 0.0) {
      native_fps = 30.0;
    }
    const double playback_fps =
        options.frame_rate != 120 ? options.frame_rate : native_fps;
    frame_interval = std::chrono::microseconds(
        static_cast<int64_t>(1000000.0 / playback_fps));
    frame_width = static_cast<uint32_t>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height =
        static_cast<uint32_t>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Video input: " << options.video_input_path << '\n'
              << "  Resolution: " << frame_width << 'x' << frame_height << '\n'
              << "  Native FPS: " << native_fps
              << ", Playback FPS: " << playback_fps << '\n'
              << "  Total frames: "
              << static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT))
              << std::endl;
  }

  bool read(CapturedFrame &frame) {
    if (!capture.read(frame.image)) {
      std::cout << "End of video reached." << std::endl;
      return false;
    }
    frame.raw_image.release();
    frame.timestamp = capture.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    return true;
  }

  void handleKey(char) {}

  cv::VideoCapture capture;
  uint32_t frame_width = 0;
  uint32_t frame_height = 0;
  std::chrono::microseconds frame_interval{0};
  std::string window_name = "video-input";
};

#else

#include "LibCamera.h"

class FrameSource::Impl {
public:
  explicit Impl(const ApplicationOptions &options)
      : frame_width(options.cam_width), frame_height(options.cam_height),
        keep_raw(options.raw_preview || options.raw_stream ||
                 options.raw_save_frame || options.raw_save_video),
        frame_interval(
            std::chrono::microseconds(1000000 / options.frame_rate)) {
    if (camera.initCamera() != 0) {
      throw std::runtime_error("Failed to initialize camera");
    }
    camera.configureStill(frame_width, frame_height, formats::R8, 1, 0);

    ControlList controls;
    const int64_t camera_frame_time = 1000000 / 120;
    controls.set(libcamera::controls::FrameDurationLimits,
                 libcamera::Span<const int64_t, 2>(
                     {camera_frame_time, camera_frame_time}));
    if (options.brightness >= -1.0 && options.brightness <= 1.0) {
      controls.set(libcamera::controls::Brightness, options.brightness);
      std::cout << "Brightness: " << options.brightness << std::endl;
    }
    if (options.contrast >= 0.0) {
      controls.set(libcamera::controls::Contrast, options.contrast);
      std::cout << "Contrast: " << options.contrast << std::endl;
    }
    if (options.exposure_time >= 0) {
      controls.set(libcamera::controls::ExposureTime, options.exposure_time);
      std::cout << "Exposure time: " << options.exposure_time << std::endl;
    }
    camera.set(controls);

    if (camera.startCamera() != 0) {
      camera.closeCamera();
      throw std::runtime_error("Failed to start camera");
    }
    started = true;
    camera.VideoStream(&frame_width, &frame_height, &stride);
  }

  ~Impl() {
    if (started) {
      camera.stopCamera();
    }
    camera.closeCamera();
  }

  bool read(CapturedFrame &frame) {
    LibcameraOutData frame_data;
    if (!camera.readFrame(&frame_data)) {
      return false;
    }

    cv::Mat raw(frame_height, frame_width, CV_16UC1, frame_data.imageData,
                stride);
    cv::Mat shifted = raw / 64;
    shifted.convertTo(frame.image, CV_8U, 255.0 / 1023.0);
    if (keep_raw) {
      frame.raw_image = frame.image.clone();
    } else {
      frame.raw_image.release();
    }

    timespec monotonic_time{}, real_time{};
    clock_gettime(CLOCK_MONOTONIC, &monotonic_time);
    clock_gettime(CLOCK_REALTIME, &real_time);
    const uint64_t monotonic_now =
        static_cast<uint64_t>(monotonic_time.tv_sec) * 1000000000ULL +
        monotonic_time.tv_nsec;
    const uint64_t realtime_now =
        static_cast<uint64_t>(real_time.tv_sec) * 1000000000ULL +
        real_time.tv_nsec;
    const int64_t monotonic_to_realtime = static_cast<int64_t>(realtime_now) -
                                          static_cast<int64_t>(monotonic_now);
    frame.timestamp =
        static_cast<double>(frame_data.timestamp + monotonic_to_realtime) /
        1000000000.0;

    camera.returnFrameBuffer(frame_data);
    return true;
  }

  void handleKey(char key) {
    if (key == 'f') {
      ControlList controls;
      controls.set(libcamera::controls::AfMode,
                   libcamera::controls::AfModeAuto);
      controls.set(libcamera::controls::AfTrigger, 0);
      camera.set(controls);
      return;
    }
    if (key != 'a' && key != 'A' && key != 'd' && key != 'D') {
      return;
    }

    lens_position += (key == 'a' || key == 'A') ? focus_step : -focus_step;
    ControlList controls;
    controls.set(libcamera::controls::AfMode,
                 libcamera::controls::AfModeManual);
    controls.set(libcamera::controls::LensPosition, lens_position);
    camera.set(controls);
  }

  LibCamera camera;
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t stride = 0;
  bool keep_raw;
  bool started = false;
  float lens_position = 100.0F;
  float focus_step = 50.0F;
  std::chrono::microseconds frame_interval;
  std::string window_name = "libcamera-demo";
};

#endif

FrameSource::FrameSource(const ApplicationOptions &options)
    : impl(std::make_unique<Impl>(options)) {}

FrameSource::~FrameSource() = default;

bool FrameSource::read(CapturedFrame &frame) { return impl->read(frame); }

void FrameSource::handleKey(char key) { impl->handleKey(key); }

uint32_t FrameSource::width() const { return impl->frame_width; }

uint32_t FrameSource::height() const { return impl->frame_height; }

std::chrono::microseconds FrameSource::frameInterval() const {
  return impl->frame_interval;
}

const std::string &FrameSource::windowName() const { return impl->window_name; }
