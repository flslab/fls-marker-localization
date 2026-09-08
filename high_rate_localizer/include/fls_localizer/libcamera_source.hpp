#pragma once

#include "fls_localizer/config.hpp"

#include <memory>
#include <opencv2/core.hpp>

namespace flsloc {

class LibcameraSource {
public:
  explicit LibcameraSource(const CameraConfig &config);
  ~LibcameraSource();
  LibcameraSource(const LibcameraSource &) = delete;
  LibcameraSource &operator=(const LibcameraSource &) = delete;

  bool read(cv::Mat &gray, double &timestamp);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flsloc
