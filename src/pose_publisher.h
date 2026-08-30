#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct AttitudeSample;
struct PoseOutput;

namespace pose_shared_memory {

inline constexpr const char *name = "/pos_shared_mem";

// Bytes 0..39 retain the original pose ABI. In grid mode they contain the
// drone position in world coordinates and its shared-attitude roll, pitch,
// and yaw. Bytes 40..79 are written by Python and read by C++ using matching
// begin/end sequence guards plus a checksum over bytes 44..75.
struct Layout {
  std::uint8_t pose_valid;
  std::uint8_t pose_padding[3];
  float x, y, z;
  float roll, pitch, yaw;
  std::uint32_t pose_tail_padding;
  double pose_timestamp;
  std::uint32_t attitude_sequence_begin;
  std::uint32_t attitude_valid;
  double attitude_timestamp;
  float qx, qy, qz, qw;
  std::uint32_t attitude_sequence_end;
  std::uint32_t attitude_checksum;
};

static_assert(sizeof(Layout) == 80);
static_assert(offsetof(Layout, x) == 4);
static_assert(offsetof(Layout, pose_timestamp) == 32);
static_assert(offsetof(Layout, attitude_sequence_begin) == 40);
static_assert(offsetof(Layout, attitude_valid) == 44);
static_assert(offsetof(Layout, attitude_timestamp) == 48);
static_assert(offsetof(Layout, qx) == 56);
static_assert(offsetof(Layout, qw) == 68);
static_assert(offsetof(Layout, attitude_sequence_end) == 72);
static_assert(offsetof(Layout, attitude_checksum) == 76);

} // namespace pose_shared_memory

class PosePublisher {
public:
  explicit PosePublisher(const char *shared_memory_name =
                             pose_shared_memory::name);
  ~PosePublisher();

  PosePublisher(const PosePublisher &) = delete;
  PosePublisher &operator=(const PosePublisher &) = delete;

  void publish(const PoseOutput &pose, double timestamp);
  bool readAttitude(double camera_timestamp, double max_age_seconds,
                    AttitudeSample &sample) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
