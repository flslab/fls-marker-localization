#include "marker_tracker.h"

#include <algorithm>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <string>

namespace {

constexpr double kMaxSyncLowBits = 2.25;
constexpr int kRequiredConfirmedDecodes = 2;
constexpr int kRequiredStaticReplacementDecodes = 3;
constexpr double kBrightIntensity = 0.8;

enum class DecoderState { WAIT_FOR_SYNC, DECODING };

} // namespace

struct MarkerTracker::TrackedBlob {
  std::uint64_t track_id;
  cv::Point2f position;
  bool active = true;
  bool visible = true;
  double last_seen_time;
  DecoderState state = DecoderState::WAIT_FOR_SYNC;
  double sync_time = 0.0;
  int bit_index = 0;
  uint16_t current_id = 0;
  bool last_state;
  double high_run_start;
  double low_run_start;
  bool sync_candidate = false;
  bool id_valid = false;
  uint16_t decoded_id = 0;
  uint16_t pending_id = 0;
  int pending_decode_count = 0;
  double last_decode_time = 0.0;

  TrackedBlob(std::uint64_t id, cv::Point2f point, double time, bool bright)
      : track_id(id), position(point), last_seen_time(time), last_state(bright),
        high_run_start(bright ? time : -1.0),
        low_run_start(bright ? -1.0 : time) {}
};

MarkerTracker::MarkerTracker(double bit_duration_ms, int payload_size,
                             double tracking_threshold, double sync_threshold,
                             bool static_markers, bool validate,
                             double dark_intensity,
                             bool retire_missing_edge_tracks)
    : bit_duration_ms_(bit_duration_ms), payload_size_(payload_size),
      tracking_threshold_(tracking_threshold),
      sync_threshold_(sync_threshold), static_markers_mode_(static_markers),
      validate_mode_(validate), dark_blob_intensity_(dark_intensity),
      retire_missing_edge_tracks_(retire_missing_edge_tracks) {}

MarkerTracker::~MarkerTracker() = default;

double MarkerTracker::bitDurationSec() const {
  return bit_duration_ms_ / 1000.0;
}

double MarkerTracker::packetDurationSec() const {
  return (payload_size_ + 6.0) * bitDurationSec();
}

void MarkerTracker::acceptStaticId(TrackedBlob &blob) {
  blob.id_valid = true;
  blob.decoded_id = 0;
  blob.pending_decode_count = 0;
  blob.pending_id = 0;
  blob.last_decode_time = 0.0;
}

void MarkerTracker::acceptDecodedId(TrackedBlob &blob, uint16_t observed_id,
                                    double current_time) {
  const double confirm_window = 2.5 * packetDurationSec();
  if (blob.pending_decode_count == 0 || blob.pending_id != observed_id ||
      current_time - blob.last_decode_time > confirm_window) {
    blob.pending_id = observed_id;
    blob.pending_decode_count = 1;
  } else {
    ++blob.pending_decode_count;
  }
  blob.last_decode_time = current_time;

  const int required_decodes =
      (observed_id != 0 && blob.id_valid && blob.decoded_id == 0)
          ? kRequiredStaticReplacementDecodes
          : kRequiredConfirmedDecodes;
  if (blob.id_valid && blob.decoded_id == observed_id) {
    blob.pending_decode_count = required_decodes;
    return;
  }
  if (blob.pending_decode_count >= required_decodes) {
    blob.decoded_id = observed_id;
    blob.id_valid = true;
    blob.pending_decode_count = 0;
  }
}

MarkerTracker::Result MarkerTracker::processFrame(cv::Mat &image,
                                                  double current_time,
                                                  double blob_area_threshold,
                                                  const std::set<std::uint64_t>
                                                      &decode_ignored_track_ids) {
  cv::Mat grayscale;
  if (image.channels() == 1) {
    grayscale = image.clone();
  } else {
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
  }
  cv::cvtColor(grayscale, image, cv::COLOR_GRAY2BGR);
  cv::GaussianBlur(grayscale, grayscale, cv::Size(3, 3), 0);

  constexpr double bright_threshold = 255.0 * kBrightIntensity;
  const double visibility_threshold =
      dark_blob_intensity_ > 0.0
          ? 255.0 * dark_blob_intensity_ * 0.5
          : bright_threshold;
  cv::Mat thresholded;
  cv::threshold(grayscale, thresholded, visibility_threshold, 255,
                cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresholded, contours, cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Point2f> current_blobs;
  std::vector<bool> current_blob_states;
  for (const auto &contour : contours) {
    const cv::Moments moments = cv::moments(contour);
    if (moments.m00 <= blob_area_threshold) {
      continue;
    }
    const int center_x = static_cast<int>(moments.m10 / moments.m00);
    const int center_y = static_cast<int>(moments.m01 / moments.m00);
    cv::circle(image, {center_x, center_y}, 10, cv::Scalar(0, 0, 255), 1);
    current_blobs.emplace_back(center_x, center_y);
    current_blob_states.push_back(
        grayscale.at<uchar>(center_y, center_x) > bright_threshold);
  }

  Result result;
  if (static_markers_mode_) {
    // Static mode supplies the ID only. Pose callers decide how many points
    // their solver requires.
    for (const auto &point : current_blobs) {
      result.current_blobs.push_back({point.x, point.y, 0});
      result.decoded_markers.push_back({point.x, point.y, 0});
      if (validate_mode_) {
        cv::circle(image, point, 8, cv::Scalar(255, 0, 0), -1);
      }
    }
    return result;
  }

  // A valid packet can stay dark for the sync bit plus an all-zero payload.
  // Retire older state before association so a new marker cannot revive it.
  const double unseen_timeout = (payload_size_ + 2.0) * bitDurationSec();
  std::vector<bool> blob_matched(current_blobs.size(), false);
  for (auto &blob : tracked_blobs_) {
    if (!blob.active) {
      continue;
    }
    if (current_time - blob.last_seen_time >= unseen_timeout) {
      blob.active = false;
      result.retired_track_ids.push_back(blob.track_id);
      continue;
    }
    blob.visible = false;

    double minimum_distance = std::numeric_limits<double>::infinity();
    int best_index = -1;
    for (std::size_t i = 0; i < current_blobs.size(); ++i) {
      if (blob_matched[i]) {
        continue;
      }
      const double distance = cv::norm(blob.position - current_blobs[i]);
      if (distance < minimum_distance) {
        minimum_distance = distance;
        best_index = static_cast<int>(i);
      }
    }

    bool current_state = false;
    if (best_index >= 0 && minimum_distance < tracking_threshold_) {
      blob.position = current_blobs[best_index];
      blob.last_seen_time = current_time;
      blob_matched[best_index] = true;
      blob.visible = true;
      current_state = current_blob_states[best_index];
    } else if (retire_missing_edge_tracks_ && dark_blob_intensity_ > 0.0 &&
               (blob.position.x <= tracking_threshold_ ||
                blob.position.y <= tracking_threshold_ ||
                blob.position.x >= image.cols - 1 - tracking_threshold_ ||
                blob.position.y >= image.rows - 1 - tracking_threshold_)) {
      // With dim-state tracking enabled, a miss is not a valid logic-0 bit.
      // Grid mode can therefore safely reacquire this cell under a new track.
      blob.active = false;
      result.retired_track_ids.push_back(blob.track_id);
      continue;
    }

    if (decode_ignored_track_ids.count(blob.track_id) != 0) {
      // Keep association and any confirmed ID, but discard partial packet
      // state so decoding restarts at a real sync edge when re-enabled.
      blob.state = DecoderState::WAIT_FOR_SYNC;
      blob.sync_time = 0.0;
      blob.bit_index = 0;
      blob.current_id = 0;
      blob.sync_candidate = false;
      blob.pending_id = 0;
      blob.pending_decode_count = 0;
      blob.last_decode_time = 0.0;
      blob.high_run_start = current_state ? current_time : -1.0;
      blob.low_run_start = current_state ? -1.0 : current_time;
      blob.last_state = current_state;
      continue;
    }

    if (blob.state == DecoderState::WAIT_FOR_SYNC) {
      if (current_state) {
        if (!blob.last_state) {
          blob.high_run_start = current_time;
          blob.low_run_start = -1.0;
          blob.sync_candidate = false;
        }

        const double static_timeout =
            (payload_size_ + 10.0) * bitDurationSec();
        if ((!blob.id_valid || blob.decoded_id == 0) &&
            blob.high_run_start >= 0.0 &&
            current_time - blob.high_run_start > static_timeout) {
          acceptStaticId(blob);
        }
      } else {
        if (blob.last_state) {
          const double high_duration = current_time - blob.high_run_start;
          const double minimum_high_for_sync =
              sync_threshold_ * bitDurationSec();
          blob.low_run_start = current_time;
          blob.sync_candidate = blob.high_run_start >= 0.0 &&
                                high_duration >= minimum_high_for_sync;
          if (blob.sync_candidate) {
            blob.state = DecoderState::DECODING;
            blob.sync_time = blob.low_run_start;
            blob.bit_index = 0;
            blob.current_id = 0;
          }
        } else if (blob.sync_candidate && blob.low_run_start >= 0.0 &&
                   current_time - blob.low_run_start >
                       kMaxSyncLowBits * bitDurationSec()) {
          blob.sync_candidate = false;
        }
        blob.high_run_start = -1.0;
      }
    } else {
      const double target_time =
          blob.sync_time + (1.5 + blob.bit_index) * bitDurationSec();
      if (current_time >= target_time) {
        blob.current_id = static_cast<uint16_t>(
            (blob.current_id << 1) | (current_state ? 1 : 0));
        ++blob.bit_index;
        if (blob.bit_index >= payload_size_) {
          acceptDecodedId(blob, blob.current_id, current_time);
          blob.state = DecoderState::WAIT_FOR_SYNC;
          blob.high_run_start = current_state ? current_time : -1.0;
          blob.low_run_start = current_state ? -1.0 : current_time;
          blob.sync_candidate = false;
        }
      }

      const double decode_timeout = (payload_size_ + 2.0) * bitDurationSec();
      if (current_time - blob.sync_time > decode_timeout) {
        blob.state = DecoderState::WAIT_FOR_SYNC;
        blob.high_run_start = current_state ? current_time : -1.0;
        blob.low_run_start = current_state ? -1.0 : current_time;
        blob.sync_candidate = false;
      }
    }

    blob.last_state = current_state;
  }

  for (std::size_t i = 0; i < current_blobs.size(); ++i) {
    if (!blob_matched[i]) {
      tracked_blobs_.emplace_back(next_track_id_++, current_blobs[i], current_time,
                                  current_blob_states[i]);
    }
  }
  tracked_blobs_.erase(
      std::remove_if(tracked_blobs_.begin(), tracked_blobs_.end(),
                     [](const TrackedBlob &blob) { return !blob.active; }),
      tracked_blobs_.end());

  for (const auto &blob : tracked_blobs_) {
    if (blob.id_valid &&
        current_time - blob.last_seen_time < unseen_timeout) {
      result.decoded_markers.push_back(
          {blob.position.x, blob.position.y,
           static_cast<int>(blob.decoded_id), blob.visible,
           current_time - blob.last_seen_time, blob.track_id});
    }
    if (blob.active && blob.visible) {
      result.current_blobs.push_back(
          {blob.position.x, blob.position.y,
           blob.id_valid ? static_cast<int>(blob.decoded_id) : -1, true, 0.0,
           blob.track_id});
    }
    if (validate_mode_ && blob.id_valid) {
      cv::circle(image, blob.position, 8, cv::Scalar(255, 0, 0), -1);
      cv::putText(image, std::to_string(blob.decoded_id),
                  {static_cast<int>(blob.position.x - 20),
                   static_cast<int>(blob.position.y - 20)},
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2,
                  cv::LINE_AA);
    }
  }
  return result;
}
