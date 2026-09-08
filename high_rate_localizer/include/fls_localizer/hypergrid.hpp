#pragma once

#include "fls_localizer/config.hpp"
#include "fls_localizer/grid_map.hpp"
#include "fls_localizer/types.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace flsloc {

class HyperGridMatcher {
public:
  HyperGridMatcher(const GridMap &map, const ApplicationConfig &config);

  std::vector<MatchedPoint>
  match(const std::vector<Blob> &blobs,
        const cv::Vec3d &prior_camera_position_world,
        const cv::Matx33d &world_to_camera_rotation) const;

private:
  std::vector<MatchedPoint>
  selectSpatially(std::vector<MatchedPoint> candidates) const;

  const GridMap &map_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_;
  std::size_t maximum_points_;
  double tolerance_cells_;
};

} // namespace flsloc
