#include "background_saver.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <utility>

BackgroundSaver::~BackgroundSaver() { stop(); }

bool BackgroundSaver::startVideo(const std::string &filename, int codec,
                                 double fps, cv::Size size, bool isColor) {
  if (video_writer.isOpened()) {
    return true;
  }
  video_filename = filename;
  video_writer.open(filename, codec, fps, size, isColor);
  return video_writer.isOpened();
}

bool BackgroundSaver::isVideoOpened() const { return video_writer.isOpened(); }

void BackgroundSaver::start() {
  if (is_running) {
    return;
  }
  is_running = true;
  worker_thread = std::thread(&BackgroundSaver::savingLoop, this);
}

void BackgroundSaver::stop() {
  if (!is_running) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    is_running = false;
  }
  task_ready.notify_all();
  if (worker_thread.joinable()) {
    worker_thread.join();
  }
  if (video_writer.isOpened()) {
    video_writer.release();
    std::cout << "Video saved to " << video_filename << std::endl;
  }
}

void BackgroundSaver::push(const cv::Mat &img, const std::string &filename,
                           bool write_video) {
  if (!is_running && !write_video && filename.empty()) {
    return;
  }

  SaveTask task{img.clone(), filename, write_video};
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    task_queue.push(std::move(task));
  }
  task_ready.notify_one();
}

void BackgroundSaver::savingLoop() {
  while (true) {
    SaveTask task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      task_ready.wait(lock,
                      [this] { return !task_queue.empty() || !is_running; });
      if (!is_running && task_queue.empty()) {
        break;
      }
      task = std::move(task_queue.front());
      task_queue.pop();
    }

    if (!task.filename.empty() && !task.img.empty()) {
      cv::imwrite(task.filename, task.img);
    }
    if (task.write_video && video_writer.isOpened() && !task.img.empty()) {
      video_writer.write(task.img);
    }
  }
}
