#pragma once

#include <cstdint>
#include <optional>

namespace flsloc {

struct SequenceProgress {
  std::optional<std::uint32_t> delta;
  bool wrapped = false;
  bool reset_or_out_of_order = false;
  bool duplicate = false;
};

// Deltas refer to previously delivered camera frames. A small modular forward
// distance across UINT32_MAX is wrap-compatible; a large backward jump is not
// interpreted as billions of dropped frames.
class FrameSequenceTracker {
public:
  SequenceProgress observe(std::uint32_t sequence) {
    SequenceProgress progress;
    if (previous_) {
      const std::uint32_t delta = sequence - *previous_;
      if (delta >= (std::uint32_t{1} << 31)) {
        progress.reset_or_out_of_order = true;
      } else {
        progress.delta = delta;
        progress.wrapped = sequence < *previous_;
        progress.duplicate = delta == 0;
      }
    }
    previous_ = sequence;
    return progress;
  }

private:
  std::optional<std::uint32_t> previous_;
};

struct FrameTiming {
  std::uint64_t buffer_timestamp_ns = 0;
  std::optional<std::int64_t> sensor_timestamp_ns;
  std::optional<std::int32_t> exposure_time_us;
  std::optional<std::int64_t> frame_duration_us;
  std::optional<double> analogue_gain;
  std::uint32_t buffer_sequence = 0;
  std::uint32_t request_sequence = 0;
  int buffer_status = 0;
  std::optional<std::uint64_t> host_completed_monotonic_ns;
  std::optional<std::uint64_t> host_read_started_monotonic_ns;
  std::optional<std::uint64_t> host_read_finished_monotonic_ns;
  std::uint64_t stale_requests_skipped = 0;
  std::uint64_t stale_requests_skipped_total = 0;
  SequenceProgress buffer_progress;
  SequenceProgress request_progress;
  bool buffer_timestamp_nonmonotonic = false;
  bool sensor_timestamp_nonmonotonic = false;
};

} // namespace flsloc
