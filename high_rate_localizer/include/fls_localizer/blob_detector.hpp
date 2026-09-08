#pragma once

#include "fls_localizer/config.hpp"
#include "fls_localizer/types.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace flsloc {

class BlobDetector {
public:
  explicit BlobDetector(DetectorConfig config);

  const std::vector<Blob> &detect(const cv::Mat &gray);

private:
  DetectorConfig config_;
  cv::Mat binary_;
  cv::Mat labels_;
  cv::Mat statistics_;
  cv::Mat centroids_;
  std::vector<Blob> blobs_;
};

} // namespace flsloc
