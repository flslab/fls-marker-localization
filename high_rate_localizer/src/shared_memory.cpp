#include "fls_localizer/shared_memory.hpp"
#include "fls_localizer/pose_solver.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace flsloc {
namespace {

template <typename Block>
std::uint32_t checksum(const Block &block, std::size_t payload_begin,
                       std::size_t payload_end) {
  constexpr std::uint32_t offset = 2166136261U;
  constexpr std::uint32_t prime = 16777619U;
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(&block);
  std::uint32_t value = offset;
  for (std::size_t index = payload_begin; index < payload_end; ++index) {
    value = (value ^ bytes[index]) * prime;
  }
  return value;
}

std::uint32_t controllerChecksum(const shared::ControllerBlock &block) {
  return checksum(block,
                  offsetof(shared::ControllerBlock, ekf_reset_generation),
                  offsetof(shared::ControllerBlock, sequence_end));
}

std::uint32_t localizerChecksum(const shared::LocalizerBlock &block) {
  return checksum(block, offsetof(shared::LocalizerBlock, pose_sequence),
                  offsetof(shared::LocalizerBlock, sequence_end));
}

} // namespace

class SharedMemory::Impl {
public:
  Impl(const std::string &name, PoseTechnique pose_technique)
      : pose_technique_(pose_technique) {
    descriptor_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0660);
    if (descriptor_ < 0 ||
        ftruncate(descriptor_, sizeof(shared::Layout)) != 0) {
      throw std::runtime_error("unable to create shared memory " + name);
    }
    void *mapping = mmap(nullptr, sizeof(shared::Layout),
                         PROT_READ | PROT_WRITE, MAP_SHARED, descriptor_, 0);
    if (mapping == MAP_FAILED) {
      throw std::runtime_error("unable to map shared memory " + name);
    }
    layout_ = static_cast<shared::Layout *>(mapping);
    if (layout_->header.magic != shared::kMagic ||
        layout_->header.abi_version != shared::kAbiVersion ||
        layout_->header.layout_size != sizeof(shared::Layout)) {
      std::memset(layout_, 0, sizeof(*layout_));
      layout_->header.magic = shared::kMagic;
      layout_->header.abi_version = shared::kAbiVersion;
      layout_->header.layout_size = sizeof(shared::Layout);
    }
  }

  ~Impl() {
    if (layout_) {
      shared::LocalizerBlock block = layout_->localizer;
      block.pose_valid = 0;
      write(block);
      munmap(layout_, sizeof(shared::Layout));
    }
    if (descriptor_ >= 0) {
      close(descriptor_);
    }
  }

  ControllerInput readController() const {
    ControllerInput result;
    shared::ControllerBlock snapshot{};
    bool stable = false;
    for (int attempt = 0; attempt < 32; ++attempt) {
      const std::uint32_t begin = __atomic_load_n(
          &layout_->controller.sequence_begin, __ATOMIC_ACQUIRE);
      if (begin == 0 || (begin & 1U) != 0U) {
        std::this_thread::yield();
        continue;
      }
      std::memcpy(&snapshot, &layout_->controller, sizeof(snapshot));
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint32_t after = __atomic_load_n(
          &layout_->controller.sequence_begin, __ATOMIC_ACQUIRE);
      stable = begin == after && begin == snapshot.sequence_end &&
               snapshot.checksum == controllerChecksum(snapshot);
      if (stable) {
        break;
      }
    }
    if (!stable) {
      return result;
    }
    result.timestamp = snapshot.timestamp;
    result.ekf_reset_generation = snapshot.ekf_reset_generation;
    result.landing_requested = snapshot.landing_requested != 0;
    result.landing_tile_i = snapshot.landing_tile_i;
    result.landing_tile_j = snapshot.landing_tile_j;
    const auto quaternion = normalizeQuaternion(
        {snapshot.qx, snapshot.qy, snapshot.qz, snapshot.qw});
    result.attitude_valid =
        snapshot.attitude_valid != 0 && quaternion.has_value();
    if (quaternion) {
      result.quaternion_xyzw = *quaternion;
    }
    return result;
  }

  void publish(const FrameResult &result) {
    const PoseSolution &pose = result.sharedMemoryPose(pose_technique_);
    shared::LocalizerBlock block{};
    block.pose_sequence = pose_sequence_;
    block.frame_id = result.frame_id;
    block.timestamp = result.timestamp;
    block.initial_pose_generation = result.initial_pose_generation;
    block.feature_count = static_cast<std::uint16_t>(result.matched.size());
    block.state = static_cast<std::uint8_t>(result.state);
    block.pose_source = static_cast<std::uint8_t>(result.source);
    block.mygrid_request = static_cast<std::uint8_t>(result.mygrid_request);
    block.pose_valid = pose.accepted ? 1 : 0;
    block.tile_i = result.tile_i;
    block.tile_j = result.tile_j;
    block.processing_ms = static_cast<float>(result.processing_ms);
    block.hypergrid_acquisition_height_m =
        result.hypergrid_acquisition_height_m;
    if (pose.accepted) {
      ++pose_sequence_;
      block.pose_sequence = pose_sequence_;
      block.x = static_cast<float>(pose.drone_position_world[0]);
      block.y = static_cast<float>(pose.drone_position_world[1]);
      block.z = static_cast<float>(pose.drone_position_world[2]);
      block.qx = static_cast<float>(pose.drone_quaternion_xyzw[0]);
      block.qy = static_cast<float>(pose.drone_quaternion_xyzw[1]);
      block.qz = static_cast<float>(pose.drone_quaternion_xyzw[2]);
      block.qw = static_cast<float>(pose.drone_quaternion_xyzw[3]);
      block.initial_yaw = static_cast<float>(pose.drone_rpy[2]);
      block.reprojection_rms = static_cast<float>(pose.reprojection_rms);
    }
    write(block);
  }

private:
  void write(shared::LocalizerBlock block) {
    const std::uint32_t current =
        __atomic_load_n(&layout_->localizer.sequence_begin, __ATOMIC_RELAXED);
    const std::uint32_t even = current == 0 ? 2U : ((current + 2U) & ~1U);
    __atomic_store_n(&layout_->localizer.sequence_begin, even - 1U,
                     __ATOMIC_RELEASE);
    block.sequence_begin = even - 1U;
    block.sequence_end = even;
    block.checksum = localizerChecksum(block);
    std::memcpy(reinterpret_cast<std::uint8_t *>(&layout_->localizer) +
                    sizeof(std::uint32_t),
                reinterpret_cast<const std::uint8_t *>(&block) +
                    sizeof(std::uint32_t),
                sizeof(block) - sizeof(std::uint32_t));
    std::atomic_thread_fence(std::memory_order_release);
    __atomic_store_n(&layout_->localizer.sequence_begin, even,
                     __ATOMIC_RELEASE);
  }

  int descriptor_ = -1;
  shared::Layout *layout_ = nullptr;
  std::uint32_t pose_sequence_ = 0;
  PoseTechnique pose_technique_ = PoseTechnique::SharedAttitude;
};

SharedMemory::SharedMemory(const std::string &name,
                           PoseTechnique pose_technique)
    : impl_(std::make_unique<Impl>(name, pose_technique)) {}
SharedMemory::~SharedMemory() = default;
ControllerInput SharedMemory::readController() const {
  return impl_->readController();
}
void SharedMemory::publish(const FrameResult &result) {
  impl_->publish(result);
}

} // namespace flsloc
