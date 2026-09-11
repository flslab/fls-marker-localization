#pragma once

#include "fls_localizer/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace flsloc::shared {

inline constexpr std::uint32_t kMagic = 0x324C5346U; // "FSL2"
inline constexpr std::uint32_t kAbiVersion = 1;

struct alignas(64) Header {
  std::uint32_t magic;
  std::uint32_t abi_version;
  std::uint32_t layout_size;
  std::uint32_t reserved_word;
  std::uint8_t reserved[48];
};

struct alignas(64) ControllerBlock {
  std::uint32_t sequence_begin;
  std::uint32_t ekf_reset_generation;
  double timestamp;
  float qx, qy, qz, qw;
  std::int32_t landing_tile_i;
  std::int32_t landing_tile_j;
  std::uint8_t attitude_valid;
  std::uint8_t landing_requested;
  std::uint8_t reserved_flags[6];
  std::uint32_t sequence_end;
  std::uint32_t checksum;
  std::uint8_t reserved[8];
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
  std::uint32_t sequence_end;
  std::uint32_t checksum;
  std::uint8_t reserved[32];
};

struct alignas(64) Layout {
  Header header;
  ControllerBlock controller;
  LocalizerBlock localizer;
};

static_assert(sizeof(Header) == 64);
static_assert(sizeof(ControllerBlock) == 64);
static_assert(sizeof(LocalizerBlock) == 128);
static_assert(sizeof(Layout) == 256);

} // namespace flsloc::shared

namespace flsloc {

class SharedMemory {
public:
  SharedMemory(const std::string &name, PoseTechnique pose_technique);
  ~SharedMemory();
  SharedMemory(const SharedMemory &) = delete;
  SharedMemory &operator=(const SharedMemory &) = delete;

  ControllerInput readController() const;
  void publish(const FrameResult &result);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flsloc
