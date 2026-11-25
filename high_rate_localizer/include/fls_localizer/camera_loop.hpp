#pragma once

#include "fls_localizer/frame_timing.hpp"
#include <chrono>
#include <opencv2/core.hpp>
#include <utility>

namespace flsloc {

// Own both luma buffers: a camera read must not overwrite the pending image.
// One-frame lookahead overlaps attitude transport with acquisition. It does
// not change exposure timestamps or allow extrapolation of attitude.
template <class Camera, class Continue, class Consume>
void runCameraLoop(Camera &camera, bool defer_alignment, Continue keep_running,
                   Consume consume) {
  using Clock = std::chrono::steady_clock;
  struct Frame {
    cv::Mat gray;
    double timestamp = 0.0;
    FrameTiming timing;
    Clock::time_point read_finished;
  };
  Frame incoming, pending;
  bool has_pending = false;
  while (keep_running()) {
    if (!camera.read(incoming.gray, incoming.timestamp, incoming.timing)) {
      // Flush a finite input, but never publish a queued frame after shutdown.
      if (has_pending && keep_running()) {
        consume(pending.gray, pending.timestamp, pending.timing,
                std::chrono::duration<double>(Clock::now() - pending.read_finished).count());
      }
      return;
    }
    incoming.read_finished = Clock::now();
    if (!keep_running()) return;
    if (!defer_alignment) {
      consume(incoming.gray, incoming.timestamp, incoming.timing, 0.0);
      continue;
    }
    if (has_pending) {
      consume(pending.gray, pending.timestamp, pending.timing,
              std::chrono::duration<double>(Clock::now() - pending.read_finished).count());
    }
    std::swap(incoming, pending);
    has_pending = true;
  }
}

} // namespace flsloc
