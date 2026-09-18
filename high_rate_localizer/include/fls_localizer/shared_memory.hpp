#pragma once

#include "fls_localizer/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace flsloc::shared {

inline constexpr std::uint32_t kMagic = 0x334C5346U; // "FSL3"
inline constexpr std::uint32_t kAbiVersion = 3;
inline constexpr std::size_t kAttitudeHistorySize = 16;

struct alignas(64) Header {
  std::uint32_t magic;
  std::uint32_t abi_version;
  std::uint32_t layout_size;
  std::uint32_t reserved_word;
  std::uint8_t reserved[48];
};

struct alignas(64) ControllerBlock {
  std::uint32_t sequence_begin;
  std::uint32_t attitude_sequence;
  std::int32_t landing_tile_i;
  std::int32_t landing_tile_j;
  std::uint8_t landing_requested;
  std::uint8_t reserved_flags[7];
  std::uint32_t sequence_end;
  std::uint32_t checksum;
  std::uint8_t reserved[32];
};

struct alignas(64) AttitudeSample {
  std::uint32_t sequence_begin;
  std::uint32_t sample_sequence;
  std::uint32_t ekf_reset_generation;
  std::uint8_t attitude_valid;
  std::uint8_t reserved_flags[3];
  double timestamp;
  float qx, qy, qz, qw;
  std::uint32_t sequence_end;
  std::uint32_t checksum;
  std::uint8_t reserved[16];
};

struct alignas(64) LocalizerBlock {
  std::uint32_t sequence_begin;
  std::uint32_t pose_sequence;
  std::uint64_t frame_id;
  double timestamp;
  float x, y, z;
  float qx, qy, qz, qw;
  float initial_yaw;
  float reprojection_rms;
  float processing_ms;
  float hypergrid_acquisition_height_m;
  std::uint32_t initial_pose_generation;
  std::uint16_t feature_count;
  std::uint8_t state;
  std::uint8_t pose_source;
  std::uint8_t mygrid_request;
  std::uint8_t pose_valid;
  std::int32_t tile_i;
  std::int32_t tile_j;
  // Synchronized FC EKF yaw minus the independent PnP yaw. This sign matches
  // yawErrorMeasurement_t in the Crazyflie firmware.
  float yaw_error;
  float pnp_reprojection_rms;
  float pnp_image_span_px;
  std::uint8_t yaw_error_valid;
  std::uint8_t yaw_error_padding[3];
  std::uint32_t sequence_end;
  std::uint32_t checksum;
  std::uint8_t reserved[16];
};

struct alignas(64) Layout {
  Header header;
  ControllerBlock controller;
  AttitudeSample attitudes[kAttitudeHistorySize];
  LocalizerBlock localizer;
};

static_assert(sizeof(Header) == 64);
static_assert(sizeof(ControllerBlock) == 64);
static_assert(offsetof(ControllerBlock, attitude_sequence) == 4);
static_assert(offsetof(ControllerBlock, landing_tile_i) == 8);
static_assert(offsetof(ControllerBlock, landing_requested) == 16);
static_assert(offsetof(ControllerBlock, sequence_end) == 24);
static_assert(offsetof(ControllerBlock, checksum) == 28);
static_assert(sizeof(AttitudeSample) == 64);
static_assert(offsetof(AttitudeSample, sample_sequence) == 4);
static_assert(offsetof(AttitudeSample, ekf_reset_generation) == 8);
static_assert(offsetof(AttitudeSample, attitude_valid) == 12);
static_assert(offsetof(AttitudeSample, timestamp) == 16);
static_assert(offsetof(AttitudeSample, qx) == 24);
static_assert(offsetof(AttitudeSample, sequence_end) == 40);
static_assert(offsetof(AttitudeSample, checksum) == 44);
static_assert(sizeof(LocalizerBlock) == 128);
static_assert(offsetof(LocalizerBlock, yaw_error) == 88);
static_assert(offsetof(LocalizerBlock, yaw_error_valid) == 100);
static_assert(offsetof(LocalizerBlock, sequence_end) == 104);
static_assert(offsetof(LocalizerBlock, checksum) == 108);
static_assert(offsetof(Layout, attitudes) == 128);
static_assert(offsetof(Layout, localizer) == 1152);
static_assert(sizeof(Layout) == 1280);

} // namespace flsloc::shared

namespace flsloc {

class SharedMemory {
public:
  SharedMemory(const std::string &name, PoseTechnique pose_technique);
  ~SharedMemory();
  SharedMemory(const SharedMemory &) = delete;
  SharedMemory &operator=(const SharedMemory &) = delete;

  ControllerInput readController(double camera_timestamp) const;
  void publish(const FrameResult &result);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flsloc
