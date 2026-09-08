#pragma once

#include <array>
#include <filesystem>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace flsloc {

struct MyGridMarker {
  int id = -1;
  int local_i = 0;
  int local_j = 0;
  cv::Point3f world{};
};

struct MyGridTile {
  int i = 0;
  int j = 0;
  cv::Point3f center{};
  std::array<int, 4> signature{};
  std::array<MyGridMarker, 4> markers{};
};

struct SignatureMatch {
  const MyGridTile *tile = nullptr;
  // observed[k] maps to tile->markers[(k + rotation) % 4].
  int rotation = 0;
};

class GridMap {
public:
  static GridMap load(const std::filesystem::path &path);

  const std::filesystem::path &path() const { return path_; }
  const cv::Point3f &origin() const { return origin_; }
  double hypergridSpacing() const { return hypergrid_spacing_; }
  double hypergridMarkerDiameter() const { return hypergrid_marker_diameter_; }
  double mygridMarkerSpacing() const { return mygrid_marker_spacing_; }
  double mygridMarkerDiameter() const { return mygrid_marker_diameter_; }
  int payloadBits() const { return payload_bits_; }
  double bitDurationSeconds() const { return bit_duration_seconds_; }
  const std::string &delimiterPattern() const { return delimiter_pattern_; }
  const std::vector<MyGridTile> &tiles() const { return tiles_; }

  cv::Point3f hypergridPoint(int grid_x, int grid_y) const;
  std::optional<SignatureMatch>
  matchSignature(const std::array<int, 4> &observed) const;
  const MyGridTile *findTile(int i, int j) const;

private:
  std::filesystem::path path_;
  cv::Point3f origin_{};
  double hypergrid_spacing_ = 0.0;
  double hypergrid_marker_diameter_ = 0.0;
  double mygrid_marker_spacing_ = 0.0;
  double mygrid_marker_diameter_ = 0.0;
  int payload_bits_ = 0;
  double bit_duration_seconds_ = 0.0;
  std::string delimiter_pattern_;
  std::vector<MyGridTile> tiles_;
};

} // namespace flsloc
