#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

using namespace cv;
using namespace std;

struct PoseResult {
    uint16_t marker_id;
    Mat tvec;
    Mat rmat;
    Vec3d roll_pitch_yaw;
    Mat marker_tvec;
    Mat marker_rmat;
    Vec3d marker_roll_pitch_yaw;
};

inline Vec3d rollPitchYawDecomposition(const Mat& rmat) {
    double yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
    double pitch = atan2(-rmat.at<double>(2, 0),
                         sqrt(pow(rmat.at<double>(2, 1), 2) + pow(rmat.at<double>(2, 2), 2)));
    double roll = atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));
    return Vec3d(roll, pitch, yaw);
}

inline vector<double> matVec3ToVector(const Mat& v) {
    return {v.at<double>(0, 0), v.at<double>(1, 0), v.at<double>(2, 0)};
}

inline vector<double> vec3dToVector(const Vec3d& v) {
    return {v[0], v[1], v[2]};
}

// Function to calculate the centroid
inline Point2f findCentroid(const vector<Point2f>& points) {
    float cx = 0, cy = 0;
    for (const auto& p : points) {
        cx += p.x;
        cy += p.y;
    }
    return {cx / points.size(), cy / points.size()};
}

// Comparator to sort points clockwise around the centroid
inline bool clockwiseComparator(const Point2f& a, const Point2f& b, const Point2f& center) {
    // Calculate angles relative to the center
    double angleA = atan2(a.y - center.y, a.x - center.x);
    double angleB = atan2(b.y - center.y, b.x - center.x);
    return angleA < angleB;  // Clockwise order
}

// Function to sort points in clockwise order
inline void sortClockwise(vector<Point2f>& points) {
    // Find the centroid
    Point2f center = findCentroid(points);

    // Sort using the comparator
    sort(points.begin(), points.end(), [&](const Point2f& a, const Point2f& b) { return clockwiseComparator(a, b, center); });
}

inline void invertPose(const cv::Mat& rvec_in, const cv::Mat& tvec_in,
                cv::Mat& rvec_out, cv::Mat& tvec_out) {
    // 1. Convert the input rotation vector to a rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec_in, R);

    // 2. Invert the rotation: The inverse of a rotation matrix is its transpose
    cv::Mat R_inv = R.t();

    // 3. Invert the translation: t_inv = -R_inv * t
    cv::Mat t_inv = -R_inv * tvec_in;

    // 4. Convert the inverse rotation matrix back to a rotation vector
    cv::Rodrigues(R_inv, rvec_out);
    tvec_out = t_inv;
}

enum class DecoderState {
    WAIT_FOR_SYNC,
    DECODING
};

struct TrackedBlob {
    Point2f position;
    bool active;
    double last_seen_time;

    DecoderState state;
    double sync_time;
    int bit_index;
    uint16_t current_id;
    bool last_state;
    double high_run_start;  // timestamp when current consecutive-high run began
    double low_run_start;   // timestamp when current consecutive-low run began
    bool sync_candidate;

    bool id_valid;
    uint16_t decoded_id;
    uint16_t pending_id;
    int pending_decode_count;
    double last_decode_time;
    double creation_time;

    TrackedBlob(Point2f p, double time)
        : position(p), active(true), last_seen_time(time), creation_time(time), state(DecoderState::WAIT_FOR_SYNC), sync_time(0), bit_index(0), current_id(0), last_state(true), high_run_start(time), low_run_start(0), sync_candidate(false), id_valid(false), decoded_id(0), pending_id(0), pending_decode_count(0), last_decode_time(0) {}
};

class MarkerTracker {
   private:
    std::vector<TrackedBlob> tracked_blobs;
    double bit_duration_ms;
    int payload_size;
    double tracking_threshold;
    double sync_threshold;  // minimum consecutive-high duration (in bit-durations) required before accepting sync
    bool static_markers_mode;

    static constexpr double MIN_SYNC_LOW_BITS = 0.45;
    static constexpr double MAX_SYNC_LOW_BITS = 2.25;
    static constexpr int REQUIRED_CONFIRMED_DECODES = 2;
    static constexpr int REQUIRED_STATIC_REPLACEMENT_DECODES = 3;

    double bitDurationSec() const {
        return bit_duration_ms / 1000.0;
    }

    double packetDurationSec() const {
        // [1 start] [0 sync] [payload] [4 rest]
        return (payload_size + 6.0) * bitDurationSec();
    }

    void acceptStaticId(TrackedBlob& tb) {
        tb.id_valid = true;
        tb.decoded_id = 0;
        tb.pending_decode_count = 0;
        tb.pending_id = 0;
        tb.last_decode_time = 0;
    }

    void acceptDecodedId(TrackedBlob& tb, uint16_t observed_id, double current_time) {
        double confirm_window = 2.5 * packetDurationSec();
        if (tb.pending_decode_count == 0 ||
            tb.pending_id != observed_id ||
            current_time - tb.last_decode_time > confirm_window) {
            tb.pending_id = observed_id;
            tb.pending_decode_count = 1;
        } else {
            tb.pending_decode_count++;
        }
        tb.last_decode_time = current_time;

        int required_decodes = (observed_id != 0 && tb.id_valid && tb.decoded_id == 0)
                                   ? REQUIRED_STATIC_REPLACEMENT_DECODES
                                   : REQUIRED_CONFIRMED_DECODES;

        if (tb.id_valid && tb.decoded_id == observed_id) {
            tb.pending_decode_count = required_decodes;
            return;
        }

        if (tb.pending_decode_count >= required_decodes) {
            tb.decoded_id = observed_id;
            tb.id_valid = true;
            tb.pending_decode_count = 0;
        }
    }

    std::vector<PoseResult> estimatePoses(Mat& im,
                                          const std::map<uint16_t, std::vector<Point2f>>& groups,
                                          const Mat& cameraMatrix, const Mat& distCoeffs,
                                          const vector<Point3f>& marker_points) const {
        std::vector<PoseResult> results;
        for (auto const& [id, pts] : groups) {
            if (pts.size() == 4 && marker_points.size() == 4) {
                std::vector<Point2f> sorted_pts = pts;
                sortClockwise(sorted_pts);

                for (int i = 0; i < 4; i++) {
                    circle(im, sorted_pts[i], 8, Scalar(0, 255, 0), -1);
                    putText(im, std::to_string(i), cv::Point(sorted_pts[i].x + 12, sorted_pts[i].y - 12),
                            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2, LINE_AA);
                }

                Point2f center = findCentroid(sorted_pts);
                putText(im, "ID: " + std::to_string(id), cv::Point(center.x - 20, center.y - 20),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2, LINE_AA);

                Mat rvec, tvec;
                bool pnp_ok = solvePnP(marker_points, sorted_pts, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_AP3P);
                if (pnp_ok && !rvec.empty()) {
                    Mat marker_rmat;
                    Rodrigues(rvec, marker_rmat);
                    Mat marker_tvec = tvec.clone();
                    Vec3d marker_roll_pitch_yaw = rollPitchYawDecomposition(marker_rmat);

                    Mat camera_rmat = marker_rmat.t();
                    Mat camera_tvec = -camera_rmat * marker_tvec;
                    Vec3d camera_roll_pitch_yaw = rollPitchYawDecomposition(camera_rmat);
                    results.push_back({id, camera_tvec, camera_rmat, camera_roll_pitch_yaw,
                                       marker_tvec, marker_rmat, marker_roll_pitch_yaw});
                }
            } else if (pts.size() < 4) {
                std::vector<Point2f> sorted_pts = pts;
                sortClockwise(sorted_pts);

                for (size_t i = 0; i < sorted_pts.size(); i++) {
                    circle(im, sorted_pts[i], 8, Scalar(0, 165, 255), -1);
                    putText(im, std::to_string(i), cv::Point(sorted_pts[i].x + 12, sorted_pts[i].y - 12),
                            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 165, 255), 2, LINE_AA);
                }

                Point2f center = findCentroid(sorted_pts);
                putText(im, "ID: " + std::to_string(id) + " (" + std::to_string(pts.size()) + "/4)", cv::Point(center.x - 20, center.y - 20),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 165, 255), 2, LINE_AA);
            }
        }

        return results;
    }

   public:
    struct BlobInfo {
        float x;
        float y;
        int id;
    };
    std::vector<BlobInfo> current_frame_blobs_info;

    MarkerTracker(double bit_duration, int payload, double tracking_thresh, double sync_thresh = 4.5, bool static_markers = false)
        : bit_duration_ms(bit_duration), payload_size(payload), tracking_threshold(tracking_thresh), sync_threshold(sync_thresh), static_markers_mode(static_markers) {}

    std::vector<PoseResult> processFrame(Mat& im, double current_time,
                                         const Mat& cameraMatrix, const Mat& distCoeffs,
                                         const vector<Point3f>& marker_points, double blob_area_threshold) {
        Mat grey_proc;
        if (im.channels() == 1)
            grey_proc = im.clone();
        else
            cvtColor(im, grey_proc, COLOR_BGR2GRAY);

        cvtColor(grey_proc, im, COLOR_GRAY2BGR);

        GaussianBlur(grey_proc, grey_proc, cv::Size(9, 9), 0);
        Mat grey;
        threshold(grey_proc, grey, 255 * 0.8, 255, THRESH_BINARY);

        vector<vector<cv::Point>> contours;
        findContours(grey, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        vector<Point2f> current_blobs;
        for (const auto& contour : contours) {
            Moments moments = cv::moments(contour);
            if (moments.m00 > blob_area_threshold) {
                int center_x = int(moments.m10 / moments.m00);
                int center_y = int(moments.m01 / moments.m00);
                circle(im, cv::Point(center_x, center_y), 10, Scalar(0, 0, 255), 1);
                current_blobs.push_back(Point2f(center_x, center_y));
            }
        }
        current_frame_blobs_info.clear();

        if (static_markers_mode) {
            std::map<uint16_t, std::vector<Point2f>> groups;
            if (current_blobs.size() == 4) {
                groups[0] = current_blobs;
                for (const auto& pt : current_blobs) {
                    current_frame_blobs_info.push_back({pt.x, pt.y, 0});
                }
            } else {
                for (const auto& pt : current_blobs) {
                    current_frame_blobs_info.push_back({pt.x, pt.y, -1});
                }
            }
            return estimatePoses(im, groups, cameraMatrix, distCoeffs, marker_points);
        }

        std::vector<bool> blob_matched(current_blobs.size(), false);
        for (auto& tb : tracked_blobs) {
            if (!tb.active)
                continue;

            double min_dist = 1e9;
            int best_idx = -1;
            for (size_t i = 0; i < current_blobs.size(); ++i) {
                if (blob_matched[i])
                    continue;
                double dist = norm(tb.position - current_blobs[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = i;
                }
            }

            bool current_state = false;
            if (best_idx != -1 && min_dist < tracking_threshold) {
                tb.position = current_blobs[best_idx];
                tb.last_seen_time = current_time;
                blob_matched[best_idx] = true;
                current_state = true;
            }

            if (tb.state == DecoderState::WAIT_FOR_SYNC) {
                if (current_state) {
                    if (!tb.last_state) {
                        double low_duration = tb.low_run_start > 0 ? current_time - tb.low_run_start : 0.0;
                        double min_low_for_sync = MIN_SYNC_LOW_BITS * bitDurationSec();
                        double max_low_for_sync = MAX_SYNC_LOW_BITS * bitDurationSec();

                        if (tb.sync_candidate &&
                            low_duration >= min_low_for_sync &&
                            low_duration <= max_low_for_sync) {
                            tb.state = DecoderState::DECODING;
                            tb.sync_time = tb.low_run_start;
                            tb.bit_index = 0;
                            tb.current_id = 0;
                            // std::cout << "[Decoder] Sync detected for blob at ("
                            //   << tb.position.x << ", " << tb.position.y << ")" << std::endl;
                        }

                        tb.high_run_start = current_time;
                        tb.low_run_start = 0;
                        tb.sync_candidate = false;
                    }

                    // Auto-detect static (always-on) markers from a sustained high run.
                    // A real blinking packet has a sync-low every packet, so its high run
                    // stays shorter than this timeout even for an all-ones payload.
                    if (tb.state == DecoderState::WAIT_FOR_SYNC && (!tb.id_valid || tb.decoded_id == 0)) {
                        double static_timeout = (payload_size + 10.0) * bitDurationSec();
                        if (tb.high_run_start > 0 && current_time - tb.high_run_start > static_timeout) {
                            acceptStaticId(tb);
                            // std::cout << "[Decoder] Auto-detected static marker 0 for blob at ("
                            //   << tb.position.x << ", " << tb.position.y << ")" << std::endl;
                        }
                    }
                } else {
                    if (tb.last_state) {
                        double high_duration = current_time - tb.high_run_start;
                        double min_high_for_sync = sync_threshold * (bit_duration_ms / 1000.0);
                        tb.low_run_start = current_time;
                        tb.sync_candidate = (tb.high_run_start > 0 && high_duration >= min_high_for_sync);
                    } else if (tb.sync_candidate && tb.low_run_start > 0) {
                        double max_low_for_sync = MAX_SYNC_LOW_BITS * bitDurationSec();
                        if (current_time - tb.low_run_start > max_low_for_sync) {
                            tb.sync_candidate = false;
                        }
                    }
                    tb.high_run_start = 0;
                }
            } else if (tb.state == DecoderState::DECODING) {
                double target_time = tb.sync_time + (1.5 + tb.bit_index) * (bit_duration_ms / 1000.0);
                if (current_time >= target_time) {
                    int bit = current_state ? 1 : 0;
                    tb.current_id = (tb.current_id << 1) | bit;
                    tb.bit_index++;
                    // std::cout << "[Decoder] Blob at (" << tb.position.x << ", " << tb.position.y
                    //           << ") bit " << tb.bit_index << "/" << payload_size
                    //           << " = " << bit << " (current ID: " << tb.current_id << ")" << std::endl;
                    if (tb.bit_index >= payload_size) {
                        acceptDecodedId(tb, tb.current_id, current_time);
                        tb.state = DecoderState::WAIT_FOR_SYNC;  // continue monitoring for re-validation
                        tb.high_run_start = current_state ? current_time : 0;
                        tb.low_run_start = current_state ? 0 : current_time;
                        tb.sync_candidate = false;
                        // std::cout << "[Decoder] Successfully decoded ID: " << tb.decoded_id
                        //           << " for blob at (" << tb.position.x << ", " << tb.position.y << ")" << std::endl;
                    }
                }
                // Timeout: if decoding takes too long, the sync was probably false — abort
                double decode_timeout = (payload_size + 2.0) * (bit_duration_ms / 1000.0);
                if (current_time - tb.sync_time > decode_timeout) {
                    tb.state = DecoderState::WAIT_FOR_SYNC;
                    tb.high_run_start = current_state ? current_time : 0;
                    tb.low_run_start = current_state ? 0 : current_time;
                    tb.sync_candidate = false;
                }
            }

            tb.last_state = current_state;

            // Timeout covers the full packet period plus margin
            double active_timeout = (payload_size + 7.0) * (bit_duration_ms / 1000.0);
            if (current_time - tb.last_seen_time > active_timeout) {
                tb.active = false;
            }
        }

        for (size_t i = 0; i < current_blobs.size(); ++i) {
            if (!blob_matched[i]) {
                tracked_blobs.push_back(TrackedBlob(current_blobs[i], current_time));
            }
        }

        tracked_blobs.erase(std::remove_if(tracked_blobs.begin(), tracked_blobs.end(),
                                           [](const TrackedBlob& tb) { return !tb.active; }),
                            tracked_blobs.end());

        std::map<uint16_t, std::vector<Point2f>> groups;
        // Hold the last known position during the entire transmission
        double group_seen_limit = (payload_size + 7.0) * (bit_duration_ms / 1000.0);
        for (const auto& tb : tracked_blobs) {
            if (tb.id_valid && (current_time - tb.last_seen_time < group_seen_limit)) {
                groups[tb.decoded_id].push_back(tb.position);
            }
        }

        for (const auto& tb : tracked_blobs) {
            if (tb.active && tb.last_seen_time == current_time) {
                current_frame_blobs_info.push_back({tb.position.x, tb.position.y, tb.id_valid ? (int)tb.decoded_id : -1});
            }
        }

        return estimatePoses(im, groups, cameraMatrix, distCoeffs, marker_points);
    }
};
