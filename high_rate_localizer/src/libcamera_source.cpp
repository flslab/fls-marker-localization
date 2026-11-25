#include "fls_localizer/libcamera_source.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace flsloc {
namespace {

std::optional<std::uint64_t> hostMonotonicNs() noexcept {
  timespec stamp{};
  if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0)
    return std::nullopt;
  return static_cast<std::uint64_t>(stamp.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(stamp.tv_nsec);
}

void measureClockPair(FrameTiming &timing) noexcept {
  // Bracket BOOTTIME by MONOTONIC reads instead of assuming identical epochs.
  // Repeat per delivered frame so a suspend/resume cannot leave a stale offset.
  const auto before = hostMonotonicNs();
  timespec boot{};
  const int status = clock_gettime(CLOCK_BOOTTIME, &boot);
  const auto after = hostMonotonicNs();
  if (!before || !after || *after < *before || status != 0)
    return;
  const std::int64_t boot_ns =
      static_cast<std::int64_t>(boot.tv_sec) * 1'000'000'000LL + boot.tv_nsec;
  timing.boottime_minus_monotonic_ns =
      boot_ns - static_cast<std::int64_t>(*before + (*after - *before) / 2);
  timing.clock_pair_uncertainty_ns = (*after - *before + 1) / 2;
}

} // namespace

class LibcameraSource::Impl {
public:
  explicit Impl(const CameraConfig &settings) : settings_(settings) {
    manager_ = std::make_unique<libcamera::CameraManager>();
    if (manager_->start() != 0 || manager_->cameras().empty()) {
      throw std::runtime_error("libcamera found no available camera");
    }
    camera_ = manager_->cameras().front();
    if (camera_->acquire() != 0) {
      throw std::runtime_error("unable to acquire libcamera device");
    }
    acquired_ = true;

    configuration_ =
        camera_->generateConfiguration({libcamera::StreamRole::Viewfinder});
    if (!configuration_ || configuration_->empty()) {
      throw std::runtime_error("unable to generate libcamera configuration");
    }
    auto &stream_config = configuration_->at(0);
    stream_config.size = libcamera::Size(settings.width, settings.height);
    if (settings.pixel_format != "YUV420") {
      throw std::runtime_error(
          "the high-rate path currently requires pixel_format YUV420");
    }
    stream_config.pixelFormat = libcamera::formats::YUV420;
    stream_config.bufferCount = settings.buffer_count;
    if (configuration_->validate() == libcamera::CameraConfiguration::Invalid) {
      throw std::runtime_error("libcamera rejected the requested stream");
    }
    if (camera_->configure(configuration_.get()) != 0) {
      throw std::runtime_error("unable to configure libcamera");
    }
    stream_ = stream_config.stream();
    width_ = stream_config.size.width;
    height_ = stream_config.size.height;
    stride_ = stream_config.stride;

    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    if (allocator_->allocate(stream_) < 0) {
      throw std::runtime_error("unable to allocate libcamera buffers");
    }
    for (const auto &buffer : allocator_->buffers(stream_)) {
      mapPlane(buffer.get());
      std::unique_ptr<libcamera::Request> request = camera_->createRequest();
      if (!request || request->addBuffer(stream_, buffer.get()) != 0) {
        throw std::runtime_error("unable to create libcamera request");
      }
      requests_.push_back(std::move(request));
    }

    camera_->requestCompleted.connect(this, &Impl::requestComplete);
    libcamera::ControlList controls(camera_->controls());
    const std::int64_t frame_duration =
        static_cast<std::int64_t>(1'000'000.0 / settings.frame_rate);
    controls.set(libcamera::controls::FrameDurationLimits,
                 libcamera::Span<const std::int64_t, 2>(
                     {frame_duration, frame_duration}));
    setManualControls(controls);
    controls.set(libcamera::controls::Brightness,
                 static_cast<float>(settings.brightness));
    controls.set(libcamera::controls::Contrast,
                 static_cast<float>(settings.contrast));
    if (camera_->start(&controls) != 0) {
      throw std::runtime_error("unable to start libcamera");
    }
    started_ = true;
    for (const auto &request : requests_) {
      setManualControls(request->controls());
      if (camera_->queueRequest(request.get()) != 0) {
        throw std::runtime_error("unable to queue libcamera request");
      }
    }
  }

  ~Impl() {
    if (started_) {
      camera_->stop();
      camera_->requestCompleted.disconnect(this, &Impl::requestComplete);
    }
    for (const auto &[buffer, mapping] : mappings_) {
      (void)buffer;
      munmap(mapping.base, mapping.length);
    }
    requests_.clear();
    allocator_.reset();
    if (acquired_) {
      camera_->release();
    }
    camera_.reset();
    if (manager_) {
      manager_->stop();
    }
  }

  bool read(cv::Mat &gray, double &timestamp, FrameTiming &timing) {
    std::uint64_t stale_since_last_read = 0;
    for (;;) {
      const auto read_started = hostMonotonicNs();
      Completed selected;
      std::deque<Completed> stale;
      {
        std::unique_lock lock(mutex_);
        if (!ready_.wait_for(lock, std::chrono::seconds(2),
                             [&] { return !completed_.empty(); })) {
          throw std::runtime_error(
              "camera produced no completed request for 2 seconds");
        }
        selected = completed_.back();
        completed_.pop_back();
        stale.swap(completed_);
      }
      for (const Completed &request : stale) {
        recycle(request.request);
      }
      stale_since_last_read += stale.size();
      stale_requests_skipped_total_ += stale.size();
      const auto iterator = selected.request->buffers().find(stream_);
      if (iterator == selected.request->buffers().end()) {
        recycle(selected.request);
        return false;
      }
      libcamera::FrameBuffer *buffer = iterator->second;
      const Mapping &mapping = mappings_.at(buffer);
      timing = selected.timing;
      measureClockPair(timing);
      timing.timestamp_offset_us = settings_.timestamp_offset_us;
      timing.capture_monotonic_ns = exposureMidpointMonotonicNs(timing);
      timing.manual_controls_match = manualCameraControlsMatch(
          timing, settings_.exposure_time_us, settings_.analogue_gain);
      // Never localize startup frames with automatic/clipped controls, missing
      // capture metadata, failed buffers, or an inconsistent clock relation.
      const bool usable =
          timing.buffer_status == 0 && *timing.manual_controls_match &&
          timing.capture_monotonic_ns && timing.host_completed_monotonic_ns &&
          *timing.capture_monotonic_ns <=
              static_cast<std::int64_t>(*timing.host_completed_monotonic_ns) &&
          static_cast<std::int64_t>(*timing.host_completed_monotonic_ns) -
                  *timing.capture_monotonic_ns <
              1'000'000'000LL;
      if (!usable) {
        recycle(selected.request);
        ++unusable_frames_skipped_total_;
        const auto now = std::chrono::steady_clock::now();
        if (!unusable_since_) {
          unusable_since_ = now;
          std::cerr
              << "waiting for camera manual controls and capture metadata: "
              << "requested_exposure_us=" << settings_.exposure_time_us
              << " requested_gain=" << settings_.analogue_gain
              << " actual_exposure_us=" << timing.exposure_time_us.value_or(-1)
              << " actual_gain=" << timing.analogue_gain.value_or(-1)
              << std::endl;
        }
        if (now - *unusable_since_ > std::chrono::seconds(2))
          throw std::runtime_error("camera controls/capture metadata failed "
                                   "validation for 2 seconds");
        continue;
      }
      unusable_since_.reset();
      timestamp = static_cast<double>(*timing.capture_monotonic_ns) * 1e-9;
      cv::Mat view(static_cast<int>(height_), static_cast<int>(width_), CV_8UC1,
                   mapping.data, stride_);
      view.copyTo(gray);
      timing.unusable_frames_skipped_total = unusable_frames_skipped_total_;
      timing.host_read_started_monotonic_ns = read_started;
      timing.host_read_finished_monotonic_ns = hostMonotonicNs();
      timing.stale_requests_skipped = stale_since_last_read;
      timing.stale_requests_skipped_total = stale_requests_skipped_total_;
      timing.buffer_progress =
          buffer_sequences_.observe(timing.buffer_sequence);
      timing.request_progress =
          request_sequences_.observe(timing.request_sequence);
      timing.buffer_timestamp_nonmonotonic =
          previous_buffer_timestamp_ &&
          timing.buffer_timestamp_ns <= *previous_buffer_timestamp_;
      previous_buffer_timestamp_ = timing.buffer_timestamp_ns;
      if (timing.sensor_timestamp_ns) {
        timing.sensor_timestamp_nonmonotonic =
            previous_sensor_timestamp_ &&
            *timing.sensor_timestamp_ns <= *previous_sensor_timestamp_;
        previous_sensor_timestamp_ = timing.sensor_timestamp_ns;
      }
      recycle(selected.request);
      return true;
    }
  }

private:
  void setManualControls(libcamera::ControlList &controls) const {
    // Supply mode and values together, including after Request::reuse() clears
    // the controls. Exposure/gain alone can be ignored while AEGC is enabled.
    controls.set(libcamera::controls::AeEnable, false);
    controls.set(libcamera::controls::ExposureTime, settings_.exposure_time_us);
    controls.set(libcamera::controls::AnalogueGain,
                 static_cast<float>(settings_.analogue_gain));
  }

  struct Completed {
    libcamera::Request *request = nullptr;
    FrameTiming timing;
  };

  struct Mapping {
    void *base = MAP_FAILED;
    std::size_t length = 0;
    std::uint8_t *data = nullptr;
  };

  void mapPlane(libcamera::FrameBuffer *buffer) {
    if (buffer->planes().empty()) {
      throw std::runtime_error("libcamera buffer has no image plane");
    }
    const auto &plane = buffer->planes().front();
    const long page_size = sysconf(_SC_PAGESIZE);
    const std::uint64_t aligned_offset =
        plane.offset & ~static_cast<std::uint64_t>(page_size - 1);
    const std::size_t delta =
        static_cast<std::size_t>(plane.offset - aligned_offset);
    const std::size_t length = delta + plane.length;
    void *base = mmap(nullptr, length, PROT_READ, MAP_SHARED, plane.fd.get(),
                      static_cast<off_t>(aligned_offset));
    if (base == MAP_FAILED) {
      throw std::runtime_error("unable to map libcamera frame buffer");
    }
    mappings_[buffer] = {base, length,
                         static_cast<std::uint8_t *>(base) + delta};
  }

  void requestComplete(libcamera::Request *request) {
    if (request->status() == libcamera::Request::RequestCancelled) {
      return;
    }
    FrameTiming timing;
    timing.host_completed_monotonic_ns = hostMonotonicNs();
    timing.request_sequence = request->sequence();
    const auto iterator = request->buffers().find(stream_);
    if (iterator != request->buffers().end()) {
      const auto &buffer_metadata = iterator->second->metadata();
      timing.buffer_timestamp_ns = buffer_metadata.timestamp;
      timing.buffer_sequence = buffer_metadata.sequence;
      timing.buffer_status = static_cast<int>(buffer_metadata.status);
    }
    const auto &metadata = request->metadata();
    if (const auto value = metadata.get(libcamera::controls::SensorTimestamp))
      timing.sensor_timestamp_ns = *value;
    if (const auto value = metadata.get(libcamera::controls::ExposureTime))
      timing.exposure_time_us = *value;
    if (const auto value = metadata.get(libcamera::controls::FrameDuration))
      timing.frame_duration_us = *value;
    if (const auto value = metadata.get(libcamera::controls::AnalogueGain))
      timing.analogue_gain = *value;
    {
      std::lock_guard lock(mutex_);
      completed_.push_back({request, timing});
    }
    ready_.notify_one();
  }

  void recycle(libcamera::Request *request) {
    request->reuse(libcamera::Request::ReuseBuffers);
    setManualControls(request->controls());
    if (camera_->queueRequest(request) != 0)
      throw std::runtime_error("unable to requeue libcamera request");
  }

  CameraConfig settings_;
  std::unique_ptr<libcamera::CameraManager> manager_;
  std::shared_ptr<libcamera::Camera> camera_;
  std::unique_ptr<libcamera::CameraConfiguration> configuration_;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
  libcamera::Stream *stream_ = nullptr;
  std::vector<std::unique_ptr<libcamera::Request>> requests_;
  std::map<libcamera::FrameBuffer *, Mapping> mappings_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Completed> completed_;
  FrameSequenceTracker buffer_sequences_;
  FrameSequenceTracker request_sequences_;
  std::optional<std::uint64_t> previous_buffer_timestamp_;
  std::optional<std::int64_t> previous_sensor_timestamp_;
  std::uint64_t stale_requests_skipped_total_ = 0;
  std::uint64_t unusable_frames_skipped_total_ = 0;
  std::optional<std::chrono::steady_clock::time_point> unusable_since_;
  unsigned int width_ = 0;
  unsigned int height_ = 0;
  unsigned int stride_ = 0;
  bool acquired_ = false;
  bool started_ = false;
};

LibcameraSource::LibcameraSource(const CameraConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}
LibcameraSource::~LibcameraSource() = default;
bool LibcameraSource::read(cv::Mat &gray, double &timestamp) {
  FrameTiming ignored;
  return impl_->read(gray, timestamp, ignored);
}
bool LibcameraSource::read(cv::Mat &gray, double &timestamp,
                           FrameTiming &timing) {
  return impl_->read(gray, timestamp, timing);
}

} // namespace flsloc
