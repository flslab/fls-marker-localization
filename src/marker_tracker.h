#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

class MarkerTracker {
public:
  struct BlobInfo {
    float x = 0.0F;
    float y = 0.0F;
    int id = -1;
    bool visible = true;
    double last_seen_age = 0.0;
    // Stable for the lifetime of a tracked blob. Consumers may associate
    // spatial metadata without changing the decoder's ID state.
    std::uint64_t track_id = 0;
  };

  struct Result {
    // Blobs observed in this frame; ID is -1 until the track is decoded.
    std::vector<BlobInfo> current_blobs;
    // Active decoded tracks, including recent temporarily invisible blobs.
    std::vector<BlobInfo> decoded_markers;
    // Tracks retired before new blobs are associated in this frame.
    std::vector<std::uint64_t> retired_track_ids;
  };

  MarkerTracker(double bit_duration_ms, int payload_size,
                double tracking_threshold, double sync_threshold = 4.5,
                bool static_markers = false, bool validate = false,
                double dark_intensity = 0.0,
                bool retire_missing_edge_tracks = false);
  ~MarkerTracker();

  MarkerTracker(const MarkerTracker &) = delete;
  MarkerTracker &operator=(const MarkerTracker &) = delete;

  Result processFrame(cv::Mat &image, double current_time,
                      double blob_area_threshold);

private:
  struct TrackedBlob;

  std::vector<TrackedBlob> tracked_blobs_;
  std::uint64_t next_track_id_ = 1;
  double bit_duration_ms_;
  int payload_size_;
  double tracking_threshold_;
  double sync_threshold_;
  bool static_markers_mode_;
  bool validate_mode_;
  double dark_blob_intensity_;
  bool retire_missing_edge_tracks_;

  double bitDurationSec() const;
  double packetDurationSec() const;
  void acceptStaticId(TrackedBlob &blob);
  void acceptDecodedId(TrackedBlob &blob, uint16_t observed_id,
                       double current_time);
};
