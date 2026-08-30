#include "frame_pacer.h"

#include <iostream>
#include <thread>

FramePacer::FramePacer(std::chrono::microseconds frame_interval,
                       int execution_time, bool print_rate)
    : frame_interval(frame_interval),
      started_at(std::chrono::steady_clock::now()),
      rate_window_started_at(started_at), next_frame_at(started_at),
      execution_limit(execution_time), print_rate(print_rate) {}

bool FramePacer::completeFrame() {
  ++frame_count;
  const auto now = std::chrono::steady_clock::now();
  if (now - rate_window_started_at >= std::chrono::seconds(1)) {
    if (print_rate) {
      std::cout << frame_count << "fps" << std::endl;
    }
    frame_count = 0;
    rate_window_started_at = now;
  }
  if (execution_limit.count() > 0 && now - started_at >= execution_limit) {
    return false;
  }

  next_frame_at += frame_interval;
  std::this_thread::sleep_until(next_frame_at);
  return true;
}
