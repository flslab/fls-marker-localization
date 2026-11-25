#include "fls_localizer/frame_timing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace flsloc {

std::optional<std::int64_t> exposureMidpointMonotonicNs(const FrameTiming &t) {
  if (!t.sensor_timestamp_ns || *t.sensor_timestamp_ns <= 0 ||
      !t.exposure_time_us || *t.exposure_time_us <= 0 ||
      !t.boottime_minus_monotonic_ns || !t.clock_pair_uncertainty_ns ||
      *t.clock_pair_uncertainty_ns > 100'000 ||
      !std::isfinite(t.timestamp_offset_us))
    return std::nullopt;
  const long double capture =
      static_cast<long double>(*t.sensor_timestamp_ns) -
      *t.boottime_minus_monotonic_ns +
      static_cast<long double>(*t.exposure_time_us) * 500.0L +
      static_cast<long double>(t.timestamp_offset_us) * 1000.0L;
  if (capture <= 0 || capture > std::numeric_limits<std::int64_t>::max())
    return std::nullopt;
  return static_cast<std::int64_t>(std::round(capture));
}

bool manualCameraControlsMatch(const FrameTiming &t, int exposure_us,
                               double gain) {
  // Permit sensor line/gain quantization, but never accept automatic settings
  // such as the observed 8089 us / 8x in place of 3000 us / 1x.
  return exposure_us > 0 && std::isfinite(gain) && gain > 0 &&
         t.exposure_time_us && *t.exposure_time_us > 0 && t.analogue_gain &&
         std::isfinite(*t.analogue_gain) && *t.analogue_gain > 0 &&
         std::abs(static_cast<double>(*t.exposure_time_us) - exposure_us) <=
             std::max(100.0, exposure_us * 0.02) &&
         std::abs(*t.analogue_gain - gain) <= std::max(0.05, gain * 0.05);
}

} // namespace flsloc
