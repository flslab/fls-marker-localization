#pragma once

#include <chrono>

class FramePacer {
public:
  FramePacer(std::chrono::microseconds frame_interval, int execution_time,
             bool print_rate);

  bool completeFrame();

private:
  std::chrono::microseconds frame_interval;
  std::chrono::steady_clock::time_point started_at;
  std::chrono::steady_clock::time_point rate_window_started_at;
  std::chrono::steady_clock::time_point next_frame_at;
  std::chrono::seconds execution_limit;
  int frame_count = 0;
  bool print_rate;
};
