#pragma once

#include "fls_localizer/grid_map.hpp"
#include "fls_localizer/types.hpp"

#include <array>
#include <deque>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace flsloc {

struct DecodedRing {
  std::array<int, 4> ids{};
  std::array<cv::Point2f, 4> image_points{};
  double score = 0.0;
};

class BlinkDecoder {
public:
  BlinkDecoder(const GridMap &map, int intensity_threshold,
               double projection_gate_px);

  std::optional<DecodedRing> update(double timestamp, const cv::Mat &gray,
                                    const std::vector<Blob> &blobs);
  bool initialized() const { return initialized_; }
  const std::array<cv::Point2f, 4> &slotCenters() const { return slots_; }
  void reset();

private:
  struct Sample {
    double timestamp = 0.0;
    std::array<bool, 4> on{};
  };

  bool initializeSlots(const std::vector<Blob> &blobs);
  void updateSlots(const std::vector<Blob> &blobs);
  std::optional<DecodedRing> decode() const;
  bool sampleAt(double timestamp, int slot, bool &on, double &time_error) const;

  const GridMap &map_;
  int intensity_threshold_;
  double projection_gate_px_;
  bool initialized_ = false;
  std::array<cv::Point2f, 4> slots_{};
  std::deque<Sample> history_;
};

} // namespace flsloc
