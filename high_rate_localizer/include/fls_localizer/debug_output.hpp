#pragma once

#include "fls_localizer/config.hpp"
#include "fls_localizer/grid_map.hpp"
#include "fls_localizer/trajectory.hpp"
#include "fls_localizer/types.hpp"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <thread>

namespace flsloc {

class DebugOutput {
public:
  DebugOutput(
      const ApplicationConfig &config, const GridMap &map,
      std::string input_description, std::string trajectory_description = {},
      std::optional<OrientationErrorConfig> orientation_error = std::nullopt);
  ~DebugOutput();
  DebugOutput(const DebugOutput &) = delete;
  DebugOutput &operator=(const DebugOutput &) = delete;

  bool wantsVideoFrame(double timestamp);
  cv::Mat annotate(const cv::Mat &image, const FrameResult &result) const;
  void submit(FrameResult result, cv::Mat annotated = {});
  void finish();

  const std::filesystem::path &logPath() const { return log_path_; }
  const std::filesystem::path &videoPath() const { return video_path_; }

private:
  struct Item {
    FrameResult result;
    cv::Mat annotated;
  };

  nlohmann::json frameJson(const FrameResult &result) const;
  nlohmann::json metadata() const;
  void worker();

  ApplicationConfig config_;
  const GridMap &map_;
  std::string input_description_;
  std::string trajectory_description_;
  std::optional<OrientationErrorConfig> orientation_error_;
  std::filesystem::path log_path_;
  std::filesystem::path temporary_log_path_;
  std::filesystem::path video_path_;
  double next_video_timestamp_ = -1.0;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable space_;
  std::deque<Item> queue_;
  std::thread worker_;
  bool stopping_ = false;
  bool finished_ = false;
  std::string worker_error_;
};

} // namespace flsloc
