#include "fls_localizer/blob_detector.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace flsloc {

BlobDetector::BlobDetector(DetectorConfig config) : config_(config) {
  blobs_.reserve(config_.maximum_candidates);
}

const std::vector<Blob> &BlobDetector::detect(const cv::Mat &gray) {
  if (gray.empty() || gray.type() != CV_8UC1) {
    throw std::invalid_argument("BlobDetector requires an 8-bit gray frame");
  }
  cv::threshold(gray, binary_, config_.intensity_threshold, 255,
                cv::THRESH_BINARY);
  const int component_count = cv::connectedComponentsWithStats(
      binary_, labels_, statistics_, centroids_, 8, CV_16U);

  blobs_.clear();
  for (int label = 1; label < component_count; ++label) {
    const int area = statistics_.at<int>(label, cv::CC_STAT_AREA);
    if (area < config_.minimum_area || area > config_.maximum_area) {
      continue;
    }
    const int left = statistics_.at<int>(label, cv::CC_STAT_LEFT);
    const int top = statistics_.at<int>(label, cv::CC_STAT_TOP);
    const int width = statistics_.at<int>(label, cv::CC_STAT_WIDTH);
    const int height = statistics_.at<int>(label, cv::CC_STAT_HEIGHT);
    if (width <= 0 || height <= 0) {
      continue;
    }
    const float aspect = static_cast<float>(width) / height;
    const float fill = static_cast<float>(area) / (width * height);
    if (aspect < 0.5F || aspect > 2.0F || fill < config_.minimum_fill_ratio) {
      continue;
    }
    Blob blob;
    blob.center = {static_cast<float>(centroids_.at<double>(label, 0)),
                   static_cast<float>(centroids_.at<double>(label, 1))};
    blob.bounds = {left, top, width, height};
    blob.area = static_cast<float>(area);
    blob.fill_ratio = fill;
    blobs_.push_back(blob);
  }

  if (blobs_.size() > config_.maximum_candidates) {
    std::nth_element(blobs_.begin(),
                     blobs_.begin() + config_.maximum_candidates, blobs_.end(),
                     [](const Blob &left, const Blob &right) {
                       return left.area > right.area;
                     });
    blobs_.resize(config_.maximum_candidates);
  }
  std::sort(blobs_.begin(), blobs_.end(),
            [](const Blob &left, const Blob &right) {
              return left.area > right.area;
            });
  return blobs_;
}

} // namespace flsloc
