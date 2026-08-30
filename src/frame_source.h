#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <string>

struct ApplicationOptions;

struct CapturedFrame {
  cv::Mat image;
  cv::Mat raw_image;
  double timestamp = 0.0;
};

class FrameSource {
public:
  explicit FrameSource(const ApplicationOptions &options);
  ~FrameSource();

  FrameSource(const FrameSource &) = delete;
  FrameSource &operator=(const FrameSource &) = delete;

  bool read(CapturedFrame &frame);
  void handleKey(char key);

  uint32_t width() const;
  uint32_t height() const;
  std::chrono::microseconds frameInterval() const;
  const std::string &windowName() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
