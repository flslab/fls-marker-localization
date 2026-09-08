#include "fls_localizer/grid_map.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>

namespace flsloc {
namespace {

using json = nlohmann::json;

cv::Point3f point3(const json &value, const char *field) {
  if (!value.is_array() || value.size() != 3) {
    throw std::runtime_error(std::string(field) + " must contain xyz");
  }
  return {value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
}

std::string canonicalRing(const std::array<int, 4> &ring) {
  std::string best;
  for (int rotation = 0; rotation < 4; ++rotation) {
    std::ostringstream candidate;
    for (int index = 0; index < 4; ++index) {
      candidate << ring[(index + rotation) % 4] << ',';
    }
    if (best.empty() || candidate.str() < best) {
      best = candidate.str();
    }
  }
  return best;
}

} // namespace

GridMap GridMap::load(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open marker grid: " + path.string());
  }
  const json root = json::parse(input);
  if (root.value("schema", "") != "fls-marker-grid") {
    throw std::runtime_error("unsupported marker-grid schema");
  }

  GridMap map;
  map.path_ = path;
  map.origin_ = point3(root.at("grid_origin"), "grid_origin");
  map.hypergrid_spacing_ = root.at("hypergrid").at("marker_spacing");
  map.hypergrid_marker_diameter_ =
      root.at("hypergrid").value("marker_diameter", 0.0);
  map.mygrid_marker_spacing_ = root.at("mygrid").at("marker_spacing");
  map.mygrid_marker_diameter_ = root.at("mygrid").value("marker_diameter", 0.0);
  map.payload_bits_ = root.at("encoding").at("payload_bits");
  map.bit_duration_seconds_ = root.at("encoding").at("bit_duration_s");
  map.delimiter_pattern_ = root.at("encoding").at("delimiter_pattern");

  if (map.hypergrid_spacing_ <= 0.0 || map.mygrid_marker_spacing_ <= 0.0 ||
      map.payload_bits_ <= 0 || map.payload_bits_ > 16 ||
      map.bit_duration_seconds_ <= 0.0 || map.delimiter_pattern_.empty()) {
    throw std::runtime_error("marker-grid geometry or encoding is invalid");
  }

  std::set<std::string> rings;
  for (const auto &tile_json : root.at("mygrid").at("tiles")) {
    MyGridTile tile;
    tile.i = tile_json.at("i");
    tile.j = tile_json.at("j");
    tile.center = point3(tile_json.at("center"), "tile.center");
    const std::vector<int> signature =
        tile_json.at("signature").get<std::vector<int>>();
    if (signature.size() != 4 || tile_json.at("markers").size() != 4) {
      throw std::runtime_error("every MyGrid tile must contain four markers");
    }
    for (int index = 0; index < 4; ++index) {
      tile.signature[index] = signature[index];
      const auto &marker_json = tile_json.at("markers").at(index);
      MyGridMarker &marker = tile.markers[index];
      marker.id = marker_json.at("id");
      marker.local_i = marker_json.at("local_row");
      marker.local_j = marker_json.at("local_col");
      marker.world =
          point3(marker_json.at("global_position"), "marker.global_position");
      if (marker.id != tile.signature[index]) {
        throw std::runtime_error("tile signature and marker IDs disagree");
      }
    }
    for (int rotation = 1; rotation < 4; ++rotation) {
      bool symmetric = true;
      for (int index = 0; index < 4; ++index) {
        symmetric = symmetric && tile.signature[index] ==
                                     tile.signature[(index + rotation) % 4];
      }
      if (symmetric) {
        throw std::runtime_error(
            "MyGrid ring signature does not determine a unique yaw");
      }
    }
    const std::string key = canonicalRing(tile.signature);
    if (!rings.insert(key).second) {
      throw std::runtime_error("MyGrid ring signatures are not unique");
    }
    map.tiles_.push_back(tile);
  }
  if (map.tiles_.empty()) {
    throw std::runtime_error("marker grid contains no MyGrid landing tiles");
  }
  return map;
}

cv::Point3f GridMap::hypergridPoint(int grid_x, int grid_y) const {
  return {origin_.x + static_cast<float>((grid_x + 0.5) * hypergrid_spacing_),
          origin_.y + static_cast<float>((grid_y + 0.5) * hypergrid_spacing_),
          origin_.z};
}

std::optional<SignatureMatch>
GridMap::matchSignature(const std::array<int, 4> &observed) const {
  std::optional<SignatureMatch> result;
  for (const MyGridTile &tile : tiles_) {
    for (int rotation = 0; rotation < 4; ++rotation) {
      bool equal = true;
      for (int index = 0; index < 4; ++index) {
        equal =
            equal && observed[index] == tile.signature[(index + rotation) % 4];
      }
      if (!equal) {
        continue;
      }
      if (result) {
        return std::nullopt;
      }
      result = SignatureMatch{&tile, rotation};
    }
  }
  return result;
}

const MyGridTile *GridMap::findTile(int i, int j) const {
  for (const MyGridTile &tile : tiles_) {
    if (tile.i == i && tile.j == j) {
      return &tile;
    }
  }
  return nullptr;
}

} // namespace flsloc
