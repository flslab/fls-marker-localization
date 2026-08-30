#pragma once

#include <condition_variable>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <thread>

class BackgroundSaver {
public:
  BackgroundSaver() = default;
  ~BackgroundSaver();

  bool startVideo(const std::string &filename, int codec, double fps,
                  cv::Size size, bool isColor);
  bool isVideoOpened() const;
  void start();
  void stop();
  void push(const cv::Mat &img, const std::string &filename, bool write_video);

private:
  struct SaveTask {
    cv::Mat img;
    std::string filename;
    bool write_video;
  };

  void savingLoop();

  std::queue<SaveTask> task_queue;
  std::mutex queue_mutex;
  std::condition_variable task_ready;
  bool is_running = false;
  std::thread worker_thread;
  cv::VideoWriter video_writer;
  std::string video_filename;
};
