#include "marker_tracker.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class TestFailure : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void check(bool condition, const char *expression, const char *file, int line) {
  if (condition) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": check failed: " << expression;
  throw TestFailure(message.str());
}

void checkNear(double actual, double expected, double tolerance,
               const char *actual_expression, const char *expected_expression,
               const char *file, int line) {
  if (std::isfinite(actual) && std::isfinite(expected) &&
      std::abs(actual - expected) <= tolerance) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": expected " << actual_expression
          << " ~= " << expected_expression << " (actual " << actual
          << ", expected " << expected << ", tolerance " << tolerance << ')';
  throw TestFailure(message.str());
}

#define CHECK(expression)                                                      \
  check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance)                                \
  checkNear((actual), (expected), (tolerance), #actual, #expected, __FILE__,   \
            __LINE__)

cv::Mat blobFrame(cv::Point center, int intensity) {
  cv::Mat frame = cv::Mat::zeros(100, 100, CV_8UC1);
  cv::circle(frame, center, 7, cv::Scalar(intensity), cv::FILLED);
  return frame;
}

void testDarkBlobIntensityKeepsTrackVisible() {
  MarkerTracker tracker(20.0, 4, 30.0, 4.5, false, true, 0.25);
  cv::Mat bright = blobFrame({40, 40}, 255);
  const MarkerTracker::Result bright_result =
      tracker.processFrame(bright, 0.0, 3.0);
  CHECK(bright_result.current_blobs.size() == 1);

  cv::Mat dark = blobFrame({43, 40}, 64);
  const MarkerTracker::Result dark_result =
      tracker.processFrame(dark, 0.01, 3.0);
  CHECK(dark_result.current_blobs.size() == 1);
  CHECK_NEAR(dark_result.current_blobs.front().x, 43.0, 1.0);

  MarkerTracker zero_dark_tracker(20.0, 4, 30.0, 4.5, false, true, 0.0);
  bright = blobFrame({40, 40}, 255);
  zero_dark_tracker.processFrame(bright, 0.0, 3.0);
  dark = blobFrame({43, 40}, 64);
  const MarkerTracker::Result zero_dark_result =
      zero_dark_tracker.processFrame(dark, 0.01, 3.0);
  CHECK(zero_dark_result.current_blobs.empty());
}

void testStaticModeDoesNotImposePoseCardinality() {
  cv::Mat frame = cv::Mat::zeros(120, 120, CV_8UC1);
  const std::vector<cv::Point> centers = {{25, 25}, {95, 25}, {25, 95}};
  for (const cv::Point center : centers) {
    cv::circle(frame, center, 7, cv::Scalar(255), cv::FILLED);
  }

  MarkerTracker tracker(20.0, 4, 30.0, 4.5, true);
  const MarkerTracker::Result result = tracker.processFrame(frame, 0.0, 3.0);

  CHECK(result.current_blobs.size() == centers.size());
  CHECK(result.decoded_markers.size() == centers.size());
  CHECK(std::all_of(result.current_blobs.begin(), result.current_blobs.end(),
                    [](const MarkerTracker::BlobInfo &blob) {
                      return blob.id == 0;
                    }));
  CHECK(std::all_of(result.decoded_markers.begin(),
                    result.decoded_markers.end(),
                    [](const MarkerTracker::BlobInfo &blob) {
                      return blob.id == 0;
                    }));
}

void testTrackIdentityIsStableAndUnique() {
  MarkerTracker tracker(20.0, 4, 30.0);
  cv::Mat first = blobFrame({30, 40}, 255);
  const MarkerTracker::Result first_result =
      tracker.processFrame(first, 0.0, 3.0);
  CHECK(first_result.current_blobs.size() == 1);
  const std::uint64_t first_track_id =
      first_result.current_blobs.front().track_id;
  CHECK(first_track_id != 0);

  cv::Mat second = cv::Mat::zeros(100, 100, CV_8UC1);
  cv::circle(second, {33, 40}, 7, cv::Scalar(255), cv::FILLED);
  cv::circle(second, {80, 70}, 7, cv::Scalar(255), cv::FILLED);
  const MarkerTracker::Result second_result =
      tracker.processFrame(second, 0.01, 3.0);
  CHECK(second_result.current_blobs.size() == 2);
  const auto continued = std::find_if(
      second_result.current_blobs.begin(), second_result.current_blobs.end(),
      [](const MarkerTracker::BlobInfo &blob) { return blob.x < 50.0F; });
  const auto added = std::find_if(
      second_result.current_blobs.begin(), second_result.current_blobs.end(),
      [](const MarkerTracker::BlobInfo &blob) { return blob.x > 50.0F; });
  CHECK(continued != second_result.current_blobs.end());
  CHECK(added != second_result.current_blobs.end());
  CHECK(continued->track_id == first_track_id);
  CHECK(added->track_id != 0);
  CHECK(added->track_id != first_track_id);
}

std::uint64_t establishStaticDecodedTrack(MarkerTracker &tracker,
                                          cv::Point center) {
  MarkerTracker::Result result;
  for (int millisecond = 0; millisecond <= 12; ++millisecond) {
    cv::Mat frame = blobFrame(center, 255);
    result = tracker.processFrame(frame, millisecond / 1000.0, 3.0);
  }
  CHECK(result.current_blobs.size() == 1);
  CHECK(result.current_blobs.front().id == 0);
  return result.current_blobs.front().track_id;
}

void testDecodeSuppressionPreservesTrackAndRestartsDecoder() {
  MarkerTracker tracker(1.0, 1, 30.0);
  cv::Mat frame = blobFrame({40, 40}, 255);
  MarkerTracker::Result result = tracker.processFrame(frame, 0.0, 3.0);
  CHECK(result.current_blobs.size() == 1);
  const std::uint64_t track_id = result.current_blobs.front().track_id;
  const std::set<std::uint64_t> ignored = {track_id};

  for (int millisecond = 1; millisecond <= 20; ++millisecond) {
    frame = blobFrame({40, 40}, 255);
    result = tracker.processFrame(frame, millisecond / 1000.0, 3.0,
                                  ignored);
    CHECK(result.current_blobs.front().track_id == track_id);
    CHECK(result.current_blobs.front().id == -1);
    CHECK(result.decoded_markers.empty());
  }

  // Suppression reset the old high interval; decoding needs a fresh full
  // static-marker interval after it is lifted.
  for (int millisecond = 21; millisecond <= 30; ++millisecond) {
    frame = blobFrame({40, 40}, 255);
    result = tracker.processFrame(frame, millisecond / 1000.0, 3.0);
    CHECK(result.current_blobs.front().track_id == track_id);
    CHECK(result.current_blobs.front().id == -1);
  }
  for (int millisecond = 31; millisecond <= 32; ++millisecond) {
    frame = blobFrame({40, 40}, 255);
    result = tracker.processFrame(frame, millisecond / 1000.0, 3.0);
  }
  CHECK(result.current_blobs.front().track_id == track_id);
  CHECK(result.current_blobs.front().id == 0);

  // A confirmed identity remains available while transient decoding is gated.
  frame = blobFrame({40, 40}, 255);
  result = tracker.processFrame(frame, 0.033, 3.0, ignored);
  CHECK(result.current_blobs.front().track_id == track_id);
  CHECK(result.current_blobs.front().id == 0);
  CHECK(result.decoded_markers.size() == 1);
  CHECK(result.decoded_markers.front().track_id == track_id);
  CHECK(result.decoded_markers.front().id == 0);
}

void testExpiredDecodedTrackIsNotReused() {
  MarkerTracker tracker(1.0, 1, 30.0);
  const std::uint64_t old_track_id =
      establishStaticDecodedTrack(tracker, {40, 40});

  cv::Mat replacement = blobFrame({42, 40}, 255);
  const MarkerTracker::Result result =
      tracker.processFrame(replacement, 0.016, 3.0);
  CHECK(result.current_blobs.size() == 1);
  CHECK(result.current_blobs.front().track_id != old_track_id);
  CHECK(result.current_blobs.front().id == -1);
  CHECK(result.decoded_markers.empty());
  CHECK(result.retired_track_ids.size() == 1);
  CHECK(result.retired_track_ids.front() == old_track_id);

  MarkerTracker cutoff_tracker(125.0, 1, 30.0);
  cv::Mat first = blobFrame({40, 40}, 255);
  const std::uint64_t cutoff_track_id =
      cutoff_tracker.processFrame(first, 0.0, 3.0)
          .current_blobs.front()
          .track_id;
  replacement = blobFrame({40, 40}, 255);
  const MarkerTracker::Result at_cutoff =
      cutoff_tracker.processFrame(replacement, 0.375, 3.0);
  CHECK(at_cutoff.current_blobs.front().track_id != cutoff_track_id);
  CHECK(at_cutoff.retired_track_ids.front() == cutoff_track_id);
}

void testGridModeRetiresMissingEdgeTrackButKeepsInteriorBlink() {
  MarkerTracker edge_tracker(20.0, 4, 30.0, 4.5, false, false, 0.25,
                             true);
  cv::Mat edge = blobFrame({8, 50}, 255);
  const MarkerTracker::Result first =
      edge_tracker.processFrame(edge, 0.0, 3.0);
  const std::uint64_t exited_track_id = first.current_blobs.front().track_id;

  cv::Mat blank = cv::Mat::zeros(100, 100, CV_8UC1);
  const MarkerTracker::Result missing =
      edge_tracker.processFrame(blank, 0.01, 3.0);
  CHECK(missing.retired_track_ids.size() == 1);
  CHECK(missing.retired_track_ids.front() == exited_track_id);

  cv::Mat entering = blobFrame({9, 50}, 255);
  const MarkerTracker::Result replacement =
      edge_tracker.processFrame(entering, 0.02, 3.0);
  CHECK(replacement.current_blobs.size() == 1);
  CHECK(replacement.current_blobs.front().track_id != exited_track_id);

  MarkerTracker edge_blink_tracker(20.0, 4, 30.0, 4.5, false, false,
                                   0.0, true);
  MarkerTracker::Result edge_blink;
  for (int frame_index = 0; frame_index <= 15; ++frame_index) {
    cv::Mat frame = blobFrame({8, 50}, 255);
    edge_blink = edge_blink_tracker.processFrame(
        frame, frame_index * 0.02, 3.0);
  }
  CHECK(edge_blink.current_blobs.front().id == 0);
  const std::uint64_t edge_blink_track_id =
      edge_blink.current_blobs.front().track_id;
  for (double timestamp : {0.32, 0.36, 0.40}) {
    blank = cv::Mat::zeros(100, 100, CV_8UC1);
    edge_blink = edge_blink_tracker.processFrame(blank, timestamp, 3.0);
    CHECK(edge_blink.retired_track_ids.empty());
  }
  edge = blobFrame({9, 50}, 255);
  edge_blink = edge_blink_tracker.processFrame(edge, 0.419, 3.0);
  CHECK(edge_blink.current_blobs.front().track_id == edge_blink_track_id);
  CHECK(edge_blink.current_blobs.front().id == 0);

  MarkerTracker interior_tracker(20.0, 4, 30.0, 4.5, false, false, 0.0,
                                 true);
  cv::Mat interior = blobFrame({50, 50}, 255);
  const MarkerTracker::Result interior_first =
      interior_tracker.processFrame(interior, 0.0, 3.0);
  const std::uint64_t interior_track_id =
      interior_first.current_blobs.front().track_id;
  blank = cv::Mat::zeros(100, 100, CV_8UC1);
  const MarkerTracker::Result interior_missing =
      interior_tracker.processFrame(blank, 0.10, 3.0);
  CHECK(interior_missing.retired_track_ids.empty());
  interior = blobFrame({52, 50}, 255);
  const MarkerTracker::Result interior_returned =
      interior_tracker.processFrame(interior, 0.119, 3.0);
  CHECK(interior_returned.current_blobs.size() == 1);
  CHECK(interior_returned.current_blobs.front().track_id == interior_track_id);
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"dark blob intensity tracking",
       testDarkBlobIntensityKeepsTrackVisible},
      {"static marker IDs do not impose pose cardinality",
       testStaticModeDoesNotImposePoseCardinality},
      {"stable unique track identity", testTrackIdentityIsStableAndUnique},
      {"decode suppression preserves and restarts track state",
       testDecodeSuppressionPreservesTrackAndRestartsDecoder},
      {"expired decoded tracks are not reused",
       testExpiredDecodedTrackIsNotReused},
      {"grid edge exits do not mix track state",
       testGridModeRetiresMissingEdgeTrackButKeepsInteriorBlink},
  };

  int failures = 0;
  for (const auto &test : tests) {
    try {
      test.second();
      std::cout << "[PASS] " << test.first << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
    }
  }
  std::cout << tests.size() - static_cast<std::size_t>(failures) << '/'
            << tests.size() << " tests passed\n";
  return failures == 0 ? 0 : 1;
}
