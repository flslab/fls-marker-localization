#include "frame_outputs.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <utility>

#include "application_options.h"
#include "background_saver.h"
#include "frame_source.h"
#include "video_streamer.h"

FrameOutputs::RateGate::RateGate(bool enabled, double rate)
    : enabled(enabled), interval(rate > 0.0 ? 1.0 / rate : 0.0) {}

bool FrameOutputs::RateGate::ready(double timestamp) {
  if (!enabled || timestamp < next_time) {
    return false;
  }
  next_time = next_time == 0.0 ? timestamp + interval : next_time + interval;
  if (next_time < timestamp) {
    next_time = timestamp + interval;
  }
  return true;
}

void FrameOutputs::RateGate::disable() { enabled = false; }

FrameOutputs::FrameOutputs(ApplicationOptions &options,
                           std::string log_directory,
                           std::string preview_window, cv::Size preview_size)
    : options(options), log_directory(std::move(log_directory)),
      preview_window(std::move(preview_window)),
      video_filename(options.video_path.empty()
                         ? this->log_directory + "/video.mp4"
                         : options.video_path),
      stream_rate(options.enable_streaming, options.stream_rate),
      image_rate(options.save_frames, options.save_rate),
      video_rate(options.save_video, options.video_fps) {
  if (options.enable_streaming) {
    streamer = std::make_unique<VideoStreamer>(options.stream_port,
                                               options.stream_type);
    if (!streamer->start()) {
      throw std::runtime_error("Failed to start video streaming");
    }
    std::cout << "Streaming at " << options.stream_rate << " fps" << std::endl;
  }

  if (options.preview) {
    cv::namedWindow(this->preview_window, cv::WINDOW_NORMAL);
    cv::resizeWindow(this->preview_window, preview_size.width,
                     preview_size.height);
  }
  if (options.save_frames) {
    const std::string directory = options.save_frames_path.empty()
                                      ? this->log_directory
                                      : options.save_frames_path;
    std::remove((directory + "/frames.zip").c_str());
    std::cout << "Saving frames at " << options.save_rate << " fps"
              << std::endl;
  }
  if (options.save_video) {
    std::remove(video_filename.c_str());
    std::cout << "Saving video at " << options.video_fps << " fps" << std::endl;
  }
  if (options.save_frames || options.save_video) {
    saver.start();
  }
}

FrameOutputs::~FrameOutputs() { finish(); }

char FrameOutputs::write(const CapturedFrame &frame, int frame_id) {
  stream(frame);
  preview(frame);
  save(frame, frame_id);
  return options.preview ? static_cast<char>(cv::waitKey(1)) : -1;
}

void FrameOutputs::finish() {
  if (finished) {
    return;
  }
  finished = true;
  saver.stop();
  if (streamer) {
    streamer->stop();
  }
  zipFrames();
  cv::destroyAllWindows();
}

double FrameOutputs::videoStartTime() const { return video_start_time; }

const cv::Mat &FrameOutputs::selectImage(bool raw,
                                         const CapturedFrame &frame) const {
  return raw && !frame.raw_image.empty() ? frame.raw_image : frame.image;
}

void FrameOutputs::stream(const CapturedFrame &frame) {
  if (!stream_rate.ready(frame.timestamp)) {
    return;
  }
  const cv::Mat &image = selectImage(options.raw_stream, frame);
  if (image.empty()) {
    return;
  }
  try {
    streamer->updateFrame(image);
  } catch (const std::exception &error) {
    std::cerr << "Streaming error: " << error.what() << std::endl;
  }
}

void FrameOutputs::preview(const CapturedFrame &frame) {
  if (options.preview) {
    cv::imshow(preview_window, selectImage(options.raw_preview, frame));
  }
}

void FrameOutputs::save(const CapturedFrame &frame, int frame_id) {
  const bool save_image = image_rate.ready(frame.timestamp);
  bool save_video = video_rate.ready(frame.timestamp);
  const cv::Mat &video_image = selectImage(options.raw_save_video, frame);
  if (save_video && !saver.isVideoOpened() && !video_image.empty()) {
    const int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    if (saver.startVideo(video_filename, codec, options.video_fps,
                         video_image.size(), video_image.channels() == 3)) {
      video_start_time = frame.timestamp;
    } else {
      std::cerr << "Could not open the output video file for write"
                << std::endl;
      options.save_video = false;
      video_rate.disable();
      save_video = false;
    }
  }
  save_video = save_video && saver.isVideoOpened();

  const cv::Mat &frame_image = selectImage(options.raw_save_frame, frame);
  const std::string frame_filename =
      save_image
          ? (options.save_frames_path.empty() ? log_directory
                                              : options.save_frames_path) +
                "/frame_" + std::to_string(frame_id) + ".jpg"
          : std::string{};

  if (save_image && save_video &&
      (&frame_image == &video_image ||
       options.raw_save_frame == options.raw_save_video)) {
    saver.push(frame_image, frame_filename, true);
    return;
  }
  if (save_image) {
    saver.push(frame_image, frame_filename, false);
  }
  if (save_video) {
    saver.push(video_image, "", true);
  }
}

void FrameOutputs::zipFrames() {
  if (!options.save_frames) {
    return;
  }
  const std::string directory = options.save_frames_path.empty()
                                    ? log_directory
                                    : options.save_frames_path;
  const std::string filename = directory + "/frames.zip";
  const std::string command =
      "zip -q -j -m " + filename + " " + directory + "/*.jpg 2>/dev/null";
  std::cout << "Zipping frames to " << filename << "..." << std::endl;
  if (std::system(command.c_str()) == 0) {
    std::cout << "Successfully zipped frames and removed originals.\n"
              << "Zip file path: " << filename << std::endl;
  } else {
    std::cerr << "Failed to zip frames or no frames found." << std::endl;
  }
}
