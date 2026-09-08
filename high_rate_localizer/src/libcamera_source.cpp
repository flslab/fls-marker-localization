#include "fls_localizer/libcamera_source.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
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
#include <unistd.h>

namespace flsloc {

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
    controls.set(libcamera::controls::ExposureTime, settings.exposure_time_us);
    controls.set(libcamera::controls::AnalogueGain,
                 static_cast<float>(settings.analogue_gain));
    controls.set(libcamera::controls::Brightness,
                 static_cast<float>(settings.brightness));
    controls.set(libcamera::controls::Contrast,
                 static_cast<float>(settings.contrast));
    if (camera_->start(&controls) != 0) {
      throw std::runtime_error("unable to start libcamera");
    }
    started_ = true;
    for (const auto &request : requests_) {
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

  bool read(cv::Mat &gray, double &timestamp) {
    libcamera::Request *selected = nullptr;
    std::deque<libcamera::Request *> stale;
    {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [&] { return !completed_.empty(); });
      selected = completed_.back();
      completed_.pop_back();
      stale.swap(completed_);
    }
    for (libcamera::Request *request : stale) {
      recycle(request);
    }
    const auto iterator = selected->buffers().find(stream_);
    if (iterator == selected->buffers().end()) {
      recycle(selected);
      return false;
    }
    libcamera::FrameBuffer *buffer = iterator->second;
    const Mapping &mapping = mappings_.at(buffer);
    const auto &metadata = buffer->metadata();
    timestamp = static_cast<double>(metadata.timestamp) * 1e-9;
    cv::Mat view(static_cast<int>(height_), static_cast<int>(width_), CV_8UC1,
                 mapping.data, stride_);
    view.copyTo(gray);
    recycle(selected);
    return true;
  }

private:
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
    {
      std::lock_guard lock(mutex_);
      completed_.push_back(request);
    }
    ready_.notify_one();
  }

  void recycle(libcamera::Request *request) {
    request->reuse(libcamera::Request::ReuseBuffers);
    camera_->queueRequest(request);
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
  std::deque<libcamera::Request *> completed_;
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
  return impl_->read(gray, timestamp);
}

} // namespace flsloc
