#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>

#include "background_saver.h"

struct ApplicationOptions;
struct CapturedFrame;
class VideoStreamer;

class FrameOutputs {
public:
  FrameOutputs(ApplicationOptions &options, std::string log_directory,
               std::string preview_window, cv::Size preview_size);
  ~FrameOutputs();

  FrameOutputs(const FrameOutputs &) = delete;
  FrameOutputs &operator=(const FrameOutputs &) = delete;

  char write(const CapturedFrame &frame, int frame_id);
  void finish();
  double videoStartTime() const;

private:
  class RateGate {
  public:
    RateGate(bool enabled, double rate);
    bool ready(double timestamp);
    void disable();

  private:
    bool enabled;
    double interval;
    double next_time = 0.0;
  };

  const cv::Mat &selectImage(bool raw, const CapturedFrame &frame) const;
  void stream(const CapturedFrame &frame);
  void preview(const CapturedFrame &frame);
  void save(const CapturedFrame &frame, int frame_id);
  void zipFrames();

  ApplicationOptions &options;
  std::string log_directory;
  std::string preview_window;
  std::string video_filename;
  std::unique_ptr<VideoStreamer> streamer;
  BackgroundSaver saver;
  RateGate stream_rate;
  RateGate image_rate;
  RateGate video_rate;
  double video_start_time = -1.0;
  bool finished = false;
};
