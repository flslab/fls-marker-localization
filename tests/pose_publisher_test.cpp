#include "frame_processor.h"
#include "pose_publisher.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint32_t attitudeChecksum(const pose_shared_memory::Layout &state) {
  constexpr std::uint32_t offset_basis = 2166136261U;
  constexpr std::uint32_t prime = 16777619U;
  constexpr std::size_t payload_offset =
      offsetof(pose_shared_memory::Layout, attitude_valid);
  constexpr std::size_t payload_size =
      offsetof(pose_shared_memory::Layout, attitude_checksum) - payload_offset;
  const auto *bytes = reinterpret_cast<const unsigned char *>(&state) +
                      payload_offset;
  std::uint32_t checksum = offset_basis;
  for (std::size_t i = 0; i < payload_size; ++i) {
    checksum = (checksum ^ bytes[i]) * prime;
  }
  return checksum;
}

void writeAttitude(pose_shared_memory::Layout &state, std::uint32_t sequence,
                   std::uint32_t valid, double timestamp,
                   const cv::Vec4f &quaternion_xyzw,
                   bool use_short_range = false) {
  require(sequence != 0 && (sequence & 1U) == 0,
          "test sequence must be nonzero and even");
  __atomic_store_n(&state.attitude_sequence_begin, sequence - 1U,
                   __ATOMIC_RELEASE);
  state.attitude_valid = valid;
  state.use_short_range = use_short_range;
  state.attitude_timestamp = timestamp;
  state.qx = quaternion_xyzw[0];
  state.qy = quaternion_xyzw[1];
  state.qz = quaternion_xyzw[2];
  state.qw = quaternion_xyzw[3];
  __atomic_store_n(&state.attitude_sequence_end, sequence, __ATOMIC_RELEASE);
  state.attitude_checksum = attitudeChecksum(state);
  __atomic_store_n(&state.attitude_sequence_begin, sequence, __ATOMIC_RELEASE);
}

} // namespace

int main() {
  const std::string name =
      "/pst" + std::to_string(static_cast<long>(getpid()));
  shm_unlink(name.c_str());

  const int probe = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (probe < 0) {
    std::cerr << "[SKIP] POSIX shared memory unavailable: "
              << std::strerror(errno) << std::endl;
    return 77;
  }
  close(probe);

  try {
    {
      PosePublisher publisher(name.c_str());
      const int descriptor = shm_open(name.c_str(), O_RDWR, 0600);
      require(descriptor >= 0, "failed to open test shared memory");
      void *mapping = mmap(nullptr, sizeof(pose_shared_memory::Layout),
                           PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
      require(mapping != MAP_FAILED, "failed to map test shared memory");
      auto &state = *static_cast<pose_shared_memory::Layout *>(mapping);

      AttitudeSample sample;
      require(!publisher.readAttitude(100.0, 0.1, sample),
              "uninitialized attitude should be absent");
      require(sample.status == "absent", "wrong uninitialized status");

      writeAttitude(state, 2, 1, 100.0, {0.0F, 0.0F, 0.0F, 4.0F}, true);
      require(state.attitude_checksum == 0x624DFAFCU,
              "attitude checksum does not match the Python wire format");
      require(publisher.readAttitude(100.02, 0.1, sample),
              "stable attitude was rejected");
      require(sample.valid && sample.status == "valid",
              "stable attitude status is invalid");
      require(sample.use_short_range,
              "shared short-range selection was not read");
      require(sample.sequence == 2, "wrong attitude sequence");
      require(std::abs(sample.age_seconds - 0.02) < 1e-12,
              "wrong attitude age");
      require(cv::norm(sample.quaternion_xyzw -
                       cv::Vec4d(0.0, 0.0, 0.0, 1.0)) < 1e-12,
              "attitude was not normalized");

      state.qx = 1.0F;
      require(!publisher.readAttitude(100.02, 0.1, sample) &&
                  sample.status == "torn",
              "checksum mismatch should be rejected");
      writeAttitude(state, 4, 1, 100.0, {0.0F, 0.0F, 0.0F, 1.0F});

      const PoseOutput pose{true, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3};
      publisher.publish(pose, 100.03);
      require(state.pose_valid == 1 && std::abs(state.x - 1.0F) < 1e-7F &&
                  std::abs(state.y - 2.0F) < 1e-7F &&
                  std::abs(state.z - 3.0F) < 1e-7F &&
                  std::abs(state.pose_timestamp - 100.03) < 1e-12,
              "legacy pose prefix changed");

      __atomic_store_n(&state.attitude_sequence_begin, 5U, __ATOMIC_RELEASE);
      require(!publisher.readAttitude(100.0, 0.1, sample) &&
                  sample.status == "torn",
              "odd sequence should be rejected");

      __atomic_store_n(&state.attitude_sequence_begin, 6U, __ATOMIC_RELEASE);
      __atomic_store_n(&state.attitude_sequence_end, 4U, __ATOMIC_RELEASE);
      require(!publisher.readAttitude(100.0, 0.1, sample) &&
                  sample.status == "torn",
              "mismatched guards should be rejected");

      writeAttitude(state, 8, 1, 100.0, {0.0F, 0.0F, 0.0F, 1.0F});
      require(!publisher.readAttitude(100.2, 0.1, sample) &&
                  sample.status == "stale",
              "stale attitude should be rejected");

      writeAttitude(state, 10, 1, 100.0,
                    {std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F,
                     1.0F});
      require(!publisher.readAttitude(100.0, 0.1, sample) &&
                  sample.status == "invalid",
              "nonfinite attitude should be rejected");

      writeAttitude(state, 12, 1, 100.0, {0.0F, 0.0F, 0.0F, 0.0F});
      require(!publisher.readAttitude(100.0, 0.1, sample) &&
                  sample.status == "invalid",
              "zero attitude should be rejected");

      writeAttitude(state, 14, 0, 100.0, {0.0F, 0.0F, 0.0F, 1.0F});
      require(!publisher.readAttitude(100.0, 0.1, sample) &&
                  sample.status == "absent",
              "invalidated attitude should be absent");

      munmap(mapping, sizeof(pose_shared_memory::Layout));
      close(descriptor);
    }
    shm_unlink(name.c_str());
    std::cout << "[PASS] shared pose/attitude snapshot" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    shm_unlink(name.c_str());
    std::cerr << "[FAIL] " << error.what() << std::endl;
    return 1;
  }
}
