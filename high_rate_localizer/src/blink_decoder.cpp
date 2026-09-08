#include "fls_localizer/blink_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace flsloc {
namespace {

std::array<cv::Point2f, 4>
clockwiseFromTopLeft(std::array<cv::Point2f, 4> points) {
  cv::Point2f center{};
  for (const cv::Point2f &point : points) {
    center += point;
  }
  center *= 0.25F;
  std::sort(points.begin(), points.end(),
            [&](const cv::Point2f &left, const cv::Point2f &right) {
              return std::atan2(left.y - center.y, left.x - center.x) <
                     std::atan2(right.y - center.y, right.x - center.x);
            });
  // atan2 starts at the left side. Rotate to the point with the smallest x+y,
  // which is the image-space top-left corner.
  int first = 0;
  for (int index = 1; index < 4; ++index) {
    if (points[index].x + points[index].y < points[first].x + points[first].y) {
      first = index;
    }
  }
  std::array<cv::Point2f, 4> ordered{};
  for (int index = 0; index < 4; ++index) {
    ordered[index] = points[(first + index) % 4];
  }
  return ordered;
}

bool squareLike(const std::array<cv::Point2f, 4> &points) {
  std::array<double, 4> sides{};
  for (int index = 0; index < 4; ++index) {
    sides[index] = cv::norm(points[index] - points[(index + 1) % 4]);
  }
  const auto [minimum, maximum] =
      std::minmax_element(sides.begin(), sides.end());
  if (*minimum < 4.0 || *maximum / *minimum > 1.8) {
    return false;
  }
  const double diagonal_a = cv::norm(points[0] - points[2]);
  const double diagonal_b = cv::norm(points[1] - points[3]);
  return std::max(diagonal_a, diagonal_b) /
             std::max(1.0, std::min(diagonal_a, diagonal_b)) <
         1.35;
}

} // namespace

BlinkDecoder::BlinkDecoder(const GridMap &map, int intensity_threshold,
                           double projection_gate_px)
    : map_(map), intensity_threshold_(intensity_threshold),
      projection_gate_px_(projection_gate_px) {}

void BlinkDecoder::reset() {
  initialized_ = false;
  history_.clear();
  slots_ = {};
}

bool BlinkDecoder::initializeSlots(const std::vector<Blob> &blobs) {
  if (blobs.size() < 4) {
    return false;
  }
  std::array<cv::Point2f, 4> points{};
  for (int index = 0; index < 4; ++index) {
    points[index] = blobs[index].center;
  }
  points = clockwiseFromTopLeft(points);
  if (!squareLike(points)) {
    return false;
  }
  slots_ = points;
  initialized_ = true;
  return true;
}

void BlinkDecoder::updateSlots(const std::vector<Blob> &blobs) {
  std::array<cv::Point2f, 4> updated = slots_;
  std::vector<bool> assigned(blobs.size(), false);
  int matched = 0;
  for (int slot = 0; slot < 4; ++slot) {
    double best = projection_gate_px_;
    int best_blob = -1;
    for (std::size_t index = 0; index < blobs.size(); ++index) {
      if (assigned[index]) {
        continue;
      }
      const double distance = cv::norm(blobs[index].center - slots_[slot]);
      if (distance < best) {
        best = distance;
        best_blob = static_cast<int>(index);
      }
    }
    if (best_blob >= 0) {
      updated[slot] = blobs[best_blob].center;
      assigned[best_blob] = true;
      ++matched;
    }
  }
  if (matched >= 3) {
    slots_ = updated;
  }
}

std::optional<DecodedRing>
BlinkDecoder::update(double timestamp, const cv::Mat &gray,
                     const std::vector<Blob> &blobs) {
  if (!initialized_ && !initializeSlots(blobs)) {
    return std::nullopt;
  }
  updateSlots(blobs);

  Sample sample;
  sample.timestamp = timestamp;
  for (int slot = 0; slot < 4; ++slot) {
    const int x = cvRound(slots_[slot].x);
    const int y = cvRound(slots_[slot].y);
    sample.on[slot] = x >= 0 && y >= 0 && x < gray.cols && y < gray.rows &&
                      gray.at<std::uint8_t>(y, x) >= intensity_threshold_;
  }
  history_.push_back(sample);
  const double history_seconds =
      (map_.payloadBits() + static_cast<int>(map_.delimiterPattern().size())) *
      map_.bitDurationSeconds() * 3.5;
  while (!history_.empty() &&
         timestamp - history_.front().timestamp > history_seconds) {
    history_.pop_front();
  }
  return decode();
}

bool BlinkDecoder::sampleAt(double timestamp, int slot, bool &on,
                            double &time_error) const {
  if (history_.empty()) {
    return false;
  }
  const Sample *best = nullptr;
  time_error = std::numeric_limits<double>::infinity();
  for (const Sample &sample : history_) {
    const double error = std::abs(sample.timestamp - timestamp);
    if (error < time_error) {
      time_error = error;
      best = &sample;
    }
  }
  if (!best || time_error > map_.bitDurationSeconds() * 0.45) {
    return false;
  }
  on = best->on[slot];
  return true;
}

std::optional<DecodedRing> BlinkDecoder::decode() const {
  const int delimiter_bits = static_cast<int>(map_.delimiterPattern().size());
  const int packet_bits = map_.payloadBits() + delimiter_bits;
  const double packet_duration = packet_bits * map_.bitDurationSeconds();
  if (history_.size() < 2 ||
      history_.back().timestamp - history_.front().timestamp <
          packet_duration * 1.8) {
    return std::nullopt;
  }

  std::optional<DecodedRing> best;
  double best_cost = std::numeric_limits<double>::infinity();
  const double search_end = history_.front().timestamp + packet_duration;
  for (const Sample &candidate : history_) {
    if (candidate.timestamp >= search_end) {
      break;
    }
    std::array<int, 4> ids{};
    int mismatch_count = 0;
    int comparison_count = 0;
    double timing_cost = 0.0;
    bool valid = true;
    for (int slot = 0; slot < 4 && valid; ++slot) {
      int reference_id = -1;
      int repetitions = 0;
      for (double start = candidate.timestamp;
           start + packet_duration <= history_.back().timestamp + 1e-9;
           start += packet_duration) {
        int id = 0;
        for (int bit = 0; bit < packet_bits; ++bit) {
          bool on = false;
          double time_error = 0.0;
          const double center = start + (bit + 0.5) * map_.bitDurationSeconds();
          bool early = false;
          bool late = false;
          double early_error = 0.0;
          double late_error = 0.0;
          const double margin = map_.bitDurationSeconds() * 0.20;
          if (!sampleAt(center, slot, on, time_error) ||
              !sampleAt(center - margin, slot, early, early_error) ||
              !sampleAt(center + margin, slot, late, late_error) ||
              early != on || late != on) {
            valid = false;
            break;
          }
          timing_cost += (time_error + early_error + late_error) /
                         map_.bitDurationSeconds();
          if (bit < map_.payloadBits()) {
            id = (id << 1) | (on ? 1 : 0);
          } else {
            const bool expected =
                map_.delimiterPattern()[bit - map_.payloadBits()] == '1';
            mismatch_count += on != expected;
            ++comparison_count;
          }
        }
        if (!valid) {
          break;
        }
        if (reference_id < 0) {
          reference_id = id;
        } else if (id != reference_id) {
          valid = false;
          break;
        }
        ++repetitions;
      }
      if (repetitions < 2) {
        valid = false;
      }
      ids[slot] = reference_id;
    }
    if (!valid || comparison_count == 0) {
      continue;
    }
    const double mismatch_rate =
        static_cast<double>(mismatch_count) / comparison_count;
    if (mismatch_rate > 0.08) {
      continue;
    }
    const double cost = mismatch_rate * 1000.0 + timing_cost;
    if (cost < best_cost) {
      best_cost = cost;
      best = DecodedRing{ids, slots_, 1.0 / (1.0 + cost)};
    }
  }
  return best;
}

} // namespace flsloc
