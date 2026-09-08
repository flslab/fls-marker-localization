#include "fls_localizer/hypergrid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <opencv2/calib3d.hpp>

namespace flsloc {

HyperGridMatcher::HyperGridMatcher(const GridMap &map,
                                   const ApplicationConfig &config)
    : map_(map), camera_matrix_(config.calibration.camera_matrix.clone()),
      distortion_(config.calibration.distortion.clone()),
      maximum_points_(config.tracking.maximum_pose_points),
      tolerance_cells_(config.tracking.lattice_tolerance_cells) {}

std::vector<MatchedPoint>
HyperGridMatcher::match(const std::vector<Blob> &blobs,
                        const cv::Vec3d &prior_camera_position_world,
                        const cv::Matx33d &world_to_camera_rotation) const {
  if (blobs.empty() || prior_camera_position_world[2] <= map_.origin().z) {
    return {};
  }
  std::vector<cv::Point2f> image_points;
  image_points.reserve(blobs.size());
  for (const Blob &blob : blobs) {
    image_points.push_back(blob.center);
  }
  std::vector<cv::Point2f> normalized;
  cv::undistortPoints(image_points, normalized, camera_matrix_, distortion_);

  std::map<std::pair<int, int>, MatchedPoint> unique;
  const double spacing = map_.hypergridSpacing();
  for (std::size_t index = 0; index < normalized.size(); ++index) {
    cv::Vec3d ray_camera(normalized[index].x, normalized[index].y, 1.0);
    const cv::Vec3d ray_world = world_to_camera_rotation.t() * ray_camera;
    if (std::abs(ray_world[2]) < 1e-8) {
      continue;
    }
    const double scale =
        (map_.origin().z - prior_camera_position_world[2]) / ray_world[2];
    if (scale <= 0.0) {
      continue;
    }
    const cv::Vec3d plane = prior_camera_position_world + scale * ray_world;
    const int grid_x = cvRound((plane[0] - map_.origin().x) / spacing - 0.5);
    const int grid_y = cvRound((plane[1] - map_.origin().y) / spacing - 0.5);
    const cv::Point3f world = map_.hypergridPoint(grid_x, grid_y);
    const double error = std::hypot(plane[0] - world.x, plane[1] - world.y);
    if (error > tolerance_cells_ * spacing) {
      continue;
    }

    // MyGrid markers are never HyperGrid correspondences. Excluding their
    // known world positions makes a failed OFF command harmless.
    bool mygrid_position = false;
    const double exclusion = std::max(0.012, map_.mygridMarkerSpacing() * 0.7);
    for (const MyGridTile &tile : map_.tiles()) {
      for (const MyGridMarker &marker : tile.markers) {
        if (std::hypot(plane[0] - marker.world.x, plane[1] - marker.world.y) <
            exclusion) {
          mygrid_position = true;
        }
      }
    }
    if (mygrid_position) {
      continue;
    }

    MatchedPoint match;
    match.blob_index = index;
    match.image = blobs[index].center;
    match.world = world;
    match.grid_x = grid_x;
    match.grid_y = grid_y;
    match.match_error = static_cast<float>(error / spacing);
    const auto key = std::make_pair(grid_x, grid_y);
    const auto existing = unique.find(key);
    if (existing == unique.end() ||
        match.match_error < existing->second.match_error) {
      unique[key] = match;
    }
  }

  std::vector<MatchedPoint> candidates;
  candidates.reserve(unique.size());
  for (auto &[key, match] : unique) {
    (void)key;
    candidates.push_back(match);
  }
  return selectSpatially(std::move(candidates));
}

std::vector<MatchedPoint>
HyperGridMatcher::selectSpatially(std::vector<MatchedPoint> candidates) const {
  if (candidates.size() <= maximum_points_) {
    return candidates;
  }
  std::vector<MatchedPoint> selected;
  selected.reserve(maximum_points_);
  auto first = std::max_element(
      candidates.begin(), candidates.end(),
      [](const MatchedPoint &left, const MatchedPoint &right) {
        return left.image.dot(left.image) < right.image.dot(right.image);
      });
  selected.push_back(*first);
  candidates.erase(first);
  while (selected.size() < maximum_points_) {
    auto best = candidates.begin();
    double best_distance = -1.0;
    for (auto candidate = candidates.begin(); candidate != candidates.end();
         ++candidate) {
      double nearest = std::numeric_limits<double>::infinity();
      for (const MatchedPoint &used : selected) {
        const cv::Point2f delta = candidate->image - used.image;
        nearest = std::min(nearest, static_cast<double>(delta.dot(delta)));
      }
      if (nearest > best_distance) {
        best_distance = nearest;
        best = candidate;
      }
    }
    selected.push_back(*best);
    candidates.erase(best);
  }
  return selected;
}

} // namespace flsloc
