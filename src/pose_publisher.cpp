#include "pose_publisher.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

#include "frame_processor.h"
#include "pose_math.h"

namespace {

using SharedState = pose_shared_memory::Layout;

std::uint32_t attitudeChecksum(const SharedState &snapshot) {
  constexpr std::uint32_t offset_basis = 2166136261U;
  constexpr std::uint32_t prime = 16777619U;
  constexpr std::size_t payload_offset =
      offsetof(SharedState, attitude_valid);
  constexpr std::size_t payload_size =
      offsetof(SharedState, attitude_checksum) - payload_offset;
  const auto *bytes = reinterpret_cast<const unsigned char *>(&snapshot) +
                      payload_offset;
  std::uint32_t checksum = offset_basis;
  for (std::size_t i = 0; i < payload_size; ++i) {
    checksum = (checksum ^ bytes[i]) * prime;
  }
  return checksum;
}

} // namespace

class PosePublisher::Impl {
public:
  explicit Impl(const char *shared_memory_name) {
    file_descriptor = shm_open(shared_memory_name, O_CREAT | O_RDWR, 0666);
    if (file_descriptor >= 0 &&
        ftruncate(file_descriptor, sizeof(SharedState)) == 0) {
      mapping = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE,
                     MAP_SHARED, file_descriptor, 0);
      if (mapping != MAP_FAILED) {
        state = static_cast<SharedState *>(mapping);
      }
    }
    if (mapping == MAP_FAILED) {
      std::cerr << "Warning: shared-memory pose output is unavailable; "
                   "continuing with process-local output"
                << std::endl;
      if (file_descriptor >= 0) {
        close(file_descriptor);
        file_descriptor = -1;
      }
    }
    *state = SharedState{};
  }

  ~Impl() {
    state->pose_valid = 0;
    if (mapping != MAP_FAILED) {
      munmap(mapping, sizeof(SharedState));
    }
    if (file_descriptor >= 0) {
      close(file_descriptor);
    }
  }

  void publish(const PoseOutput &pose, double timestamp) {
    state->pose_valid = pose.valid ? 1 : 0;
    state->pose_timestamp = timestamp;
    if (!pose.valid) {
      return;
    }
    state->x = pose.x;
    state->y = pose.y;
    state->z = pose.z;
    state->roll = pose.roll;
    state->pitch = pose.pitch;
    state->yaw = pose.yaw;
  }

  bool readAttitude(double camera_timestamp, double max_age_seconds,
                    AttitudeSample &sample) const {
    sample = AttitudeSample{};
    sample.status = "absent";

    SharedState snapshot{};
    std::uint32_t begin = 0;
    bool stable_snapshot = false;
    constexpr int max_snapshot_attempts = 8;
    for (int attempt = 0; attempt < max_snapshot_attempts; ++attempt) {
      begin = __atomic_load_n(&state->attitude_sequence_begin,
                              __ATOMIC_ACQUIRE);
      if (begin == 0) {
        return false;
      }
      if ((begin & 1U) == 0U) {
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(&snapshot, state, sizeof(snapshot));
        std::atomic_thread_fence(std::memory_order_acquire);
        const std::uint32_t begin_after = __atomic_load_n(
            &state->attitude_sequence_begin, __ATOMIC_ACQUIRE);
        stable_snapshot =
            begin == begin_after &&
            begin == snapshot.attitude_sequence_begin &&
            begin == snapshot.attitude_sequence_end &&
            attitudeChecksum(snapshot) == snapshot.attitude_checksum;
        if (stable_snapshot) {
          break;
        }
      }
      std::this_thread::yield();
    }
    if (!stable_snapshot) {
      sample.status = "torn";
      return false;
    }

    sample.sequence = begin;
    sample.host_timestamp = snapshot.attitude_timestamp;
    if (snapshot.attitude_valid == 0U) {
      return false;
    }
    if (!std::isfinite(camera_timestamp) ||
        !std::isfinite(snapshot.attitude_timestamp) ||
        !std::isfinite(max_age_seconds) || max_age_seconds <= 0.0) {
      sample.status = "invalid";
      return false;
    }

    sample.age_seconds =
        std::abs(camera_timestamp - snapshot.attitude_timestamp);
    if (sample.age_seconds > max_age_seconds) {
      sample.status = "stale";
      return false;
    }

    const auto normalized = pose_math::normalizeQuaternionXyzw(
        {snapshot.qx, snapshot.qy, snapshot.qz, snapshot.qw});
    if (!normalized) {
      sample.status = "invalid";
      return false;
    }
    sample.quaternion_xyzw = *normalized;
    sample.valid = true;
    sample.status = "valid";
    return true;
  }

private:
  SharedState local_state{};
  SharedState *state = &local_state;
  int file_descriptor = -1;
  void *mapping = MAP_FAILED;
};

PosePublisher::PosePublisher(const char *shared_memory_name)
    : impl(std::make_unique<Impl>(shared_memory_name)) {}

PosePublisher::~PosePublisher() = default;

void PosePublisher::publish(const PoseOutput &pose, double timestamp) {
  impl->publish(pose, timestamp);
}

bool PosePublisher::readAttitude(double camera_timestamp,
                                 double max_age_seconds,
                                 AttitudeSample &sample) const {
  return impl->readAttitude(camera_timestamp, max_age_seconds, sample);
}
