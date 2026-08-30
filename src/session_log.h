#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

struct ApplicationOptions;
class FrameProcessor;

std::string createSessionDirectory(const ApplicationOptions &options);

class SessionLog {
public:
  explicit SessionLog(const ApplicationOptions &options);

  void addFrame(int frame_id, double timestamp, nlohmann::json frame_log);
  void save(const std::string &log_directory, const FrameProcessor &processor,
            cv::Size image_size, double video_start_time) const;

private:
  const ApplicationOptions &options;
  std::vector<nlohmann::json> frames;
};
