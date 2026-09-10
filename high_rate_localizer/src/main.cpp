#include "fls_localizer/config.hpp"
#include "fls_localizer/debug_output.hpp"
#include "fls_localizer/grid_map.hpp"
#include "fls_localizer/libcamera_source.hpp"
#include "fls_localizer/pipeline.hpp"
#include "fls_localizer/shared_memory.hpp"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::atomic<bool> running{true};
void stop(int) { running.store(false); }

struct Arguments {
  std::filesystem::path config;
  std::string tag;
};

Arguments parse(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value after " + option);
    }
    if (option == "--config") {
      result.config = argv[++index];
    } else if (option == "--tag") {
      result.tag = argv[++index];
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.config.empty()) {
    throw std::runtime_error(
        "usage: fls_localizer --config FILE [--tag TAG]");
  }
  return result;
}

} // namespace

int main(int argc, char **argv) try {
  const Arguments arguments = parse(argc, argv);
  flsloc::ApplicationConfig config =
      flsloc::loadApplicationConfig(arguments.config);
  flsloc::applyOutputTag(config.output, arguments.tag);
  flsloc::GridMap map = flsloc::GridMap::load(config.grid_file);
  flsloc::LocalizationPipeline pipeline(config, map);
  flsloc::LibcameraSource camera(config.camera);
  flsloc::SharedMemory shared_memory(config.shared_memory_name);
  flsloc::DebugOutput output(config, map, "libcamera");
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);

  cv::Mat gray;
  double timestamp = 0.0;
  std::uint64_t frame_id = 0;
  while (running.load() && camera.read(gray, timestamp)) {
    const flsloc::ControllerInput controller = shared_memory.readController();
    flsloc::FrameResult result =
        pipeline.process(frame_id, timestamp, gray, controller);
    shared_memory.publish(result);
    cv::Mat annotated;
    if (output.wantsVideoFrame(timestamp)) {
      annotated = output.annotate(gray, result);
    }
    output.submit(std::move(result), std::move(annotated));
    ++frame_id;
  }
  output.finish();
  std::cout << "log=" << output.logPath() << '\n'
            << "annotated_video=" << output.videoPath() << std::endl;
  return 0;
} catch (const std::exception &error) {
  std::cerr << "fls_localizer: " << error.what() << std::endl;
  return 1;
}
