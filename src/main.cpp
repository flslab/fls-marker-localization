#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <thread>
#include <vector>

#include "LibCamera.h"
#include "aruco_tracker.h"
#include "position_kalman_filter.h"
#include "marker_tracker.h"
#include "video_streamer.h"

using namespace cv;
using namespace std;
using json = nlohmann::json;

std::atomic<bool> keep_running(true);

void signalHandler(int signum) {
    keep_running = false;
}


struct Position {
    bool valid;
    float x, y, z;
    float roll, pitch, yaw;
    double timestamp;
};


// Background saver class
class BackgroundSaver {
   private:
    struct SaveTask {
        Mat img;
        string filename;
        bool write_video;
    };

    std::queue<SaveTask> task_queue;
    std::mutex queue_mutex;
    std::condition_variable cv_;
    bool is_running;
    std::thread worker_thread;
    cv::VideoWriter video_writer;
    string video_filename;

   public:
    BackgroundSaver() : is_running(false) {}

    ~BackgroundSaver() {
        stop();
    }

    bool startVideo(const string& filename, int codec, double fps, cv::Size size, bool isColor) {
        if (video_writer.isOpened())
            return true;
        video_filename = filename;
        video_writer.open(filename, codec, fps, size, isColor);
        return video_writer.isOpened();
    }

    bool isVideoOpened() const {
        return video_writer.isOpened();
    }

    void start() {
        if (is_running)
            return;
        is_running = true;
        worker_thread = std::thread(&BackgroundSaver::savingLoop, this);
    }

    void stop() {
        if (!is_running)
            return;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            is_running = false;
        }
        cv_.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        if (video_writer.isOpened()) {
            video_writer.release();
            cout << "Video saved to " << video_filename << endl;
        }
    }

    void push(const Mat& img, const string& filename, bool write_video) {
        if (!is_running && !write_video && filename.empty())
            return;

        SaveTask task;
        // MUST deep copy the image because the original buffer will be reused by libcamera
        task.img = img.clone();
        task.filename = filename;
        task.write_video = write_video;

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
        }
        cv_.notify_one();
    }

   private:
    void savingLoop() {
        while (true) {
            SaveTask task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv_.wait(lock, [this] { return !task_queue.empty() || !is_running; });

                if (!is_running && task_queue.empty()) {
                    break;
                }

                task = task_queue.front();
                task_queue.pop();
            }

            if (!task.filename.empty() && !task.img.empty()) {
                imwrite(task.filename, task.img);
            }
            if (task.write_video && video_writer.isOpened() && !task.img.empty()) {
                video_writer.write(task.img);
            }
        }
    }
};

// Function to get a timestamped filename
std::string generateLogName() {
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    std::ostringstream filename;
    filename << "logs/"
             << std::put_time(localTime, "%H_%M_%S_%m_%d_%Y");

    return filename.str();
}

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH unknown
#endif

#define XSTR(x) STR(x)
#define STR(x) #x

std::string getGitCommitHash() {
    return XSTR(GIT_COMMIT_HASH);
}


bool readConfigFile(const string& filename, Mat& cameraMatrix, Mat& distCoeffs, vector<Point3f>& marker_points) {
    try {
        // Read JSON file
        std::ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open config file: " << filename << endl;
            return false;
        }
        json config = json::parse(file);

        // Read camera matrix
        cameraMatrix = Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cameraMatrix.at<double>(i, j) = config["/camera_matrix"_json_pointer][i][j];
            }
        }

        // Read distortion coefficients
        vector<double> dist = config["dist_coeffs"];
        distCoeffs = Mat(dist, true);

        // Read marker points
        marker_points.clear();
        if (config.contains("marker_points")) {
            for (const auto& point : config["marker_points"]) {
                marker_points.push_back(Point3f(point[0], point[1], point[2]));
            }
        }

        return true;
    } catch (const exception& e) {
        cerr << "Error reading config file: " << e.what() << endl;
        return false;
    }
}

/**
 * Read the "aruco_markers" section from the config and build the
 * data structures needed to construct an ArucoTracker.
 */
bool readArucoConfig(const string& filename,
                     string& aruco_dictionary,
                     double& aruco_marker_size,
                     std::map<int, ArucoTracker::MarkerWorldPose>& known_markers) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open config file: " << filename << endl;
            return false;
        }
        json config = json::parse(file);

        if (!config.contains("aruco_markers")) {
            cerr << "Config file does not contain 'aruco_markers' section" << endl;
            return false;
        }

        const auto& ac = config["aruco_markers"];
        aruco_dictionary = ac.value("dictionary", "DICT_4X4_50");
        aruco_marker_size = ac.value("marker_size", 0.20);

        known_markers.clear();
        if (ac.contains("markers")) {
            for (auto it = ac["markers"].begin(); it != ac["markers"].end(); ++it) {
                int marker_id = std::stoi(it.key());
                const auto& m = it.value();

                // Position (x, y, z) in world frame
                double px = m["position"][0].get<double>();
                double py = m["position"][1].get<double>();
                double pz = m["position"][2].get<double>();

                // Rotation (roll, pitch, yaw) in degrees
                double roll_deg = m["rotation_deg"][0].get<double>();
                double pitch_deg = m["rotation_deg"][1].get<double>();
                double yaw_deg = m["rotation_deg"][2].get<double>();

                double deg2rad = CV_PI / 180.0;
                cv::Mat R = eulerToRotationMatrix(roll_deg * deg2rad,
                                                  pitch_deg * deg2rad,
                                                  yaw_deg * deg2rad);
                cv::Mat t = (cv::Mat_<double>(3, 1) << px, py, pz);
                cv::Mat T = makeTransform(R, t);

                ArucoTracker::MarkerWorldPose mwp;
                mwp.id = marker_id;
                mwp.T_world_marker = T;
                known_markers[marker_id] = mwp;

                cout << "  ArUco marker " << marker_id
                     << " at [" << px << ", " << py << ", " << pz << "]" << endl;
            }
        }

        cout << "ArUco config: dictionary=" << aruco_dictionary
             << ", marker_size=" << aruco_marker_size << "m"
             << ", " << known_markers.size() << " known markers" << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error reading ArUco config: " << e.what() << endl;
        return false;
    }
}

bool createDirectory(const string& dir) {
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        // Directory does not exist, create it
        return mkdir(dir.c_str(), 0777) == 0;
    }
    return S_ISDIR(info.st_mode);  // Check if it's a directory
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    bool print_logs = false;
    bool preview = false;
    double distance = -1.0;
    int execution_time = 0;
    double save_rate = 1.0;
    bool save_frames = false;
    string save_frames_path = "";
    bool save_video = false;
    int video_fps = 30;
    string video_path = "";
    string config_file = "camera_config.json";
    string json_path = "";
    bool raw_preview = false;
    bool raw_stream = false;
    bool raw_save_frame = false;
    bool raw_save_video = false;

    // ArUco mode
    bool aruco_mode = false;

    // Streaming parameters
    bool enable_streaming = false;
    int stream_port = 8080;
    string stream_type = "http";  // "http" or "udp"
    double stream_rate = 10.0;

    double contrast = -2.0;
    double brightness = -2.0;
    int exposure_time = -2;
    int frame_rate = 120;
    int encoder_frame_rate = 50;
    int cam_width = 640;
    int cam_height = 400;

    double blob_area_threshold = 3;
    int payload_size = 4;
    int target_id = -1;
    double tracking_threshold = 30.0;
    double sync_threshold = 4.5;  // minimum consecutive-high duration (in bit-durations) for sync
    bool static_markers_mode = false;

    // Kalman filter parameters
    bool enable_kalman_filter = false;
    double kf_process_noise = 0.5;
    double kf_measurement_noise = 0.02;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "--verbose" || arg == "-v") {
            print_logs = true;
        } else if (arg == "--preview" || arg == "-p") {
            preview = true;
        } else if ((arg == "--distance" || arg == "-d") && i + 1 < argc) {
            try {
                distance = stod(argv[++i]);
                if (distance <= 0) {
                    throw invalid_argument("Distance must be positive");
                }
            } catch (const invalid_argument& e) {
                cerr << "Invalid value for distance. Must be a positive number." << endl;
                return -1;
            }
        } else if ((arg == "--time" || arg == "-t") && i + 1 < argc) {
            try {
                execution_time = stoi(argv[++i]);
            } catch (const invalid_argument& e) {
                cerr << "Invalid value for time. Must be a number." << endl;
                return -1;
            }
        } else if ((arg == "--save-frames")) {
            save_frames = true;
        } else if ((arg == "--save-frames-path") && i + 1 < argc) {
            save_frames_path = argv[++i];
        } else if ((arg == "--raw-preview")) {
            raw_preview = true;
        } else if ((arg == "--raw-stream")) {
            raw_stream = true;
        } else if ((arg == "--raw-save-frame")) {
            raw_save_frame = true;
        } else if ((arg == "--raw-save-video")) {
            raw_save_video = true;
        } else if ((arg == "--save-video" || arg == "-s")) {
            save_video = true;
        } else if ((arg == "--video-fps") && i + 1 < argc) {
            video_fps = stoi(argv[++i]);
        } else if ((arg == "--video-path") && i + 1 < argc) {
            video_path = argv[++i];
        } else if ((arg == "--json-path") && i + 1 < argc) {
            json_path = argv[++i];
        } else if ((arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        } else if ((arg == "--save-rate") && i + 1 < argc) {
            save_rate = stod(argv[++i]);
        } else if ((arg == "--contrast") && i + 1 < argc) {
            contrast = stod(argv[++i]);
        } else if ((arg == "--brightness") && i + 1 < argc) {
            brightness = stod(argv[++i]);
        } else if ((arg == "--exposure") && i + 1 < argc) {
            exposure_time = stoi(argv[++i]);
        } else if ((arg == "--fps") && i + 1 < argc) {
            frame_rate = stoi(argv[++i]);
        } else if ((arg == "--encoder-fps") && i + 1 < argc) {
            encoder_frame_rate = stoi(argv[++i]);
        } else if ((arg == "--stream") || arg == "--streaming") {
            enable_streaming = true;
        } else if ((arg == "--stream-port") && i + 1 < argc) {
            stream_port = stoi(argv[++i]);
        } else if ((arg == "--stream-type") && i + 1 < argc) {
            stream_type = argv[++i];
            if (stream_type != "http" && stream_type != "udp") {
                cerr << "Invalid stream type. Use 'http' or 'udp'." << endl;
                return -1;
            }
        } else if ((arg == "--stream-rate") && i + 1 < argc) {
            stream_rate = stod(argv[++i]);
        } else if ((arg == "--blob-area-threshold") && i + 1 < argc) {
            blob_area_threshold = stod(argv[++i]);
        } else if ((arg == "--payload-size") && i + 1 < argc) {
            payload_size = stoi(argv[++i]);
        } else if ((arg == "--target-id") && i + 1 < argc) {
            target_id = stoi(argv[++i]);
        } else if ((arg == "--tracking-threshold") && i + 1 < argc) {
            tracking_threshold = stod(argv[++i]);
        } else if ((arg == "--sync-threshold") && i + 1 < argc) {
            sync_threshold = stod(argv[++i]);
        } else if (arg == "--static-markers" || arg == "--static-blobs") {
            static_markers_mode = true;
        } else if (arg == "--aruco") {
            aruco_mode = true;
        } else if ((arg == "--width") && i + 1 < argc) {
            cam_width = stoi(argv[++i]);
        } else if ((arg == "--height") && i + 1 < argc) {
            cam_height = stoi(argv[++i]);
        } else if (arg == "--kalman-filter" || arg == "--kf") {
            enable_kalman_filter = true;
        } else if ((arg == "--kf-process-noise") && i + 1 < argc) {
            kf_process_noise = stod(argv[++i]);
        } else if ((arg == "--kf-measurement-noise") && i + 1 < argc) {
            kf_measurement_noise = stod(argv[++i]);
        }
    }

    cout << "Running at " << frame_rate << " Hz" << endl;
    cout << "Decoding marker IDs at " << encoder_frame_rate << " Hz" << endl;
    if (static_markers_mode) {
        cout << "Static marker blob mode ENABLED" << endl;
    }
    if (enable_kalman_filter) {
        cout << "Kalman filter ENABLED  (process_noise=" << kf_process_noise
             << ", measurement_noise=" << kf_measurement_noise << ")" << endl;
    }

    string log_dir = generateLogName();
    if (!createDirectory(log_dir)) {
        cerr << "Error: Unable to create directory " << log_dir << endl;
        return -1;
    }

    if (!save_frames_path.empty()) {
        if (!createDirectory(save_frames_path)) {
            cerr << "Error: Unable to create directory " << save_frames_path << endl;
            return -1;
        }
    }

    // Initialize video streamer
    VideoStreamer* streamer = nullptr;
    if (enable_streaming) {
        streamer = new VideoStreamer(stream_port, stream_type);
        if (!streamer->start()) {
            cerr << "Failed to start video streaming" << endl;
            delete streamer;
            return -1;
        }
        cout << "Streaming at " << stream_rate << " fps" << endl;
    }

    double image_save_fps = save_rate > 0 ? save_rate : 0;
    if (save_frames) {
        cout << "Saving frames at " << image_save_fps << " fps" << endl;
    }

    if (save_video) {
        cout << "Saving video at " << video_fps << " fps" << endl;
    }

    string video_filename = video_path.empty() ? (log_dir + "/video.mp4") : video_path;
    double video_start_time = -1.0;

    double video_interval = save_video && video_fps > 0 ? (1.0 / video_fps) : 0;
    double video_next_time = 0;

    double image_interval = save_frames && image_save_fps > 0 ? (1.0 / image_save_fps) : 0;
    double image_next_time = 0;

    double stream_interval = enable_streaming && stream_rate > 0 ? (1.0 / stream_rate) : 0;
    double stream_next_time = 0;

    BackgroundSaver saver;
    if (save_frames || save_video) {
        saver.start();
    }

    time_t start_time = time(0);
    int frame_count = 0;
    int frameCount = 0;
    int elapsed_seconds = 0;
    float lens_position = 100;
    float focus_step = 50;
    MarkerTracker tracker(3000.0 / encoder_frame_rate, payload_size, tracking_threshold, sync_threshold, static_markers_mode);
    LibCamera cam;
    uint32_t width = cam_width;
    uint32_t height = cam_height;
    uint32_t stride;
    char key;
    int window_width = cam_width;
    int window_height = cam_height;

    if (preview) {
        if (width > window_width) {
            cv::namedWindow("libcamera-demo", cv::WINDOW_NORMAL);
            cv::resizeWindow("libcamera-demo", window_width, window_height);
        }
    }

    int ret = cam.initCamera();
    cam.configureStill(width, height, formats::R8, 1, 0);
    ControlList controls_;
    // int64_t frame_time = 1000000 / frame_rate;
    int64_t frame_time = 1000000 / 120;
    controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_time, frame_time}));
    // Frame pacing is handled by sleep_until in the main loop,
    // not by FrameDurationLimits, to avoid constraining exposure time.
    if (brightness >= -1.0 && brightness <= 1.0) {
        controls_.set(controls::Brightness, brightness);
        cout << "Brightness: " << brightness << endl;
    }

    if (contrast >= 0.0) {
        controls_.set(controls::Contrast, contrast);
        cout << "Contrast: " << contrast << endl;
    }

    if (exposure_time >= 0) {
        controls_.set(controls::ExposureTime, exposure_time);
        cout << "Exposure time: " << exposure_time << endl;
    }

    cam.set(controls_);

    Mat cameraMatrix, distCoeffs;
    vector<Point3f> marker_points;

    if (!readConfigFile(config_file, cameraMatrix, distCoeffs, marker_points)) {
        cerr << "Failed to read camera configuration" << endl;
        return -1;
    }

    // ArUco tracker (only initialised when --aruco is active)
    ArucoTracker* aruco_tracker = nullptr;
    std::map<int, ArucoTracker::MarkerWorldPose> known_markers;
    if (aruco_mode) {
        string aruco_dictionary;
        double aruco_marker_size;

        if (!readArucoConfig(config_file, aruco_dictionary, aruco_marker_size, known_markers)) {
            cerr << "Failed to read ArUco configuration" << endl;
            return -1;
        }
        if (known_markers.empty()) {
            cerr << "No ArUco markers defined in config" << endl;
            return -1;
        }
        aruco_tracker = new ArucoTracker(aruco_dictionary, aruco_marker_size, known_markers);
        cout << "ArUco detection mode ENABLED" << endl;
    }

    if (!ret) {
        bool flag;
        LibcameraOutData frameData;
        cam.startCamera();
        cam.VideoStream(&width, &height, &stride);
        std::vector<json> frames;

        // Per-marker Kalman filters (created on demand)
        std::map<uint16_t, PositionKalmanFilter> kalman_filters;

        const char* shm_name = "/pos_shared_mem";
        int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
        ftruncate(shm_fd, sizeof(Position));
        Position* pos = (Position*)mmap(0, sizeof(Position), PROT_WRITE, MAP_SHARED, shm_fd, 0);

        // Frame pacing: process at frame_rate without affecting camera exposure
        auto frame_interval = std::chrono::microseconds(1000000 / frame_rate);
        auto next_frame_time = std::chrono::steady_clock::now();

        while (keep_running) {
            flag = cam.readFrame(&frameData);
            if (!flag)
                continue;

            // Create a properly aligned and continuous Mat from camera data
            Mat raw_frame(height, width, CV_16UC1, frameData.imageData, stride);
            cv::Mat shifted = raw_frame / 64;  // assumes raw_frame is CV_16UC1
            shifted.convertTo(raw_frame, CV_8U, 255.0 / 1023.0);
            //            cvtColor(raw_frame, raw_frame, cv::COLOR_GRAY2BGR);
            Mat im;

            // Ensure the frame is continuous and properly aligned
            if (raw_frame.isContinuous() && stride == width * 3) {
                im = raw_frame.clone();  // Safe copy
            } else {
                // Create a properly aligned copy
                raw_frame.copyTo(im);
            }

            Mat raw_im;
            if (raw_preview || raw_stream || raw_save_frame || raw_save_video) {
                raw_im = im.clone();
            }

            struct timespec ts_mono, ts_real;
            clock_gettime(CLOCK_MONOTONIC, &ts_mono);
            clock_gettime(CLOCK_REALTIME, &ts_real);
            uint64_t mono_now = (uint64_t)ts_mono.tv_sec * 1000000000ULL + ts_mono.tv_nsec;
            uint64_t real_now = (uint64_t)ts_real.tv_sec * 1000000000ULL + ts_real.tv_nsec;
            int64_t mono_to_real_offset = (int64_t)real_now - (int64_t)mono_now;

            uint64_t camera_realtime_ns = frameData.timestamp + mono_to_real_offset;
            double current_time_sec = (double)camera_realtime_ns / 1000000000.0;

            // Process the image
            std::vector<json> current_frame_poses;
            std::vector<json> blob_entries;
            bool pos_updated = false;

            if (aruco_mode && aruco_tracker) {
                // ── ArUco detection mode ─────────────────────────────
                // Convert to colour so annotations and saved frames are in colour
                if (im.channels() == 1) {
                    cvtColor(im, im, COLOR_GRAY2BGR);
                }
                auto aruco_result = aruco_tracker->processFrame(im, cameraMatrix, distCoeffs);

                if (aruco_result.valid) {
                    std::vector<double> camera_position_vec = matVec3ToVector(aruco_result.tvec_world);
                    std::vector<double> camera_orientation_vec = vec3dToVector(aruco_result.roll_pitch_yaw);
                    std::vector<json> marker_pose_entries;
                    for (int marker_id : aruco_result.detected_ids) {
                        auto it = known_markers.find(marker_id);
                        if (it == known_markers.end()) {
                            continue;
                        }

                        const Mat& T_world_marker = it->second.T_world_marker;
                        Mat marker_rmat = T_world_marker(cv::Rect(0, 0, 3, 3));
                        Mat marker_tvec = T_world_marker(cv::Rect(3, 0, 1, 3));
                        Vec3d marker_rpy = rollPitchYawDecomposition(marker_rmat);
                        marker_pose_entries.push_back({
                            {"marker_pose", true},
                            {"marker_id", marker_id},
                            {"marker_position", matVec3ToVector(marker_tvec)},
                            {"marker_orientation", vec3dToVector(marker_rpy)},
                        });
                    }

                    if (print_logs) {
                        cout << "[ArUco] Camera world pos: ["
                             << camera_position_vec[0] << ", " << camera_position_vec[1] << ", " << camera_position_vec[2]
                             << "]  RPY: ["
                             << camera_orientation_vec[0] << ", " << camera_orientation_vec[1] << ", " << camera_orientation_vec[2] << "]"
                             << "  markers: " << aruco_result.markers_used
                             << "  reproj_err: " << aruco_result.reprojection_error << endl;
                    }

                    json pose_entry = {
                        {"camera_pose", true},
                        {"camera_position", camera_position_vec},
                        {"camera_orientation", camera_orientation_vec},
                        {"markers_used", aruco_result.markers_used},
                        {"detected_ids", aruco_result.detected_ids},
                        {"reprojection_error", aruco_result.reprojection_error}};

                    // Kalman-filtered position (uses marker_id 0 slot for camera pose)
                    std::vector<double> filtered_vec = camera_position_vec;
                    if (enable_kalman_filter) {
                        const uint16_t kf_id = 0xFFFF;  // reserved KF slot for ArUco camera pose
                        if (kalman_filters.find(kf_id) == kalman_filters.end()) {
                            kalman_filters.emplace(kf_id,
                                                   PositionKalmanFilter(kf_process_noise, kf_measurement_noise));
                        }
                        Eigen::Vector3d meas(camera_position_vec[0], camera_position_vec[1], camera_position_vec[2]);
                        Eigen::Vector3d filt = kalman_filters.at(kf_id).update(meas, current_time_sec);
                        filtered_vec = {filt[0], filt[1], filt[2]};
                        pose_entry["camera_position_filtered"] = filtered_vec;
                    }

                    current_frame_poses.push_back(pose_entry);
                    current_frame_poses.insert(current_frame_poses.end(),
                                               marker_pose_entries.begin(), marker_pose_entries.end());

                    // Shared memory
                    pos->valid = true;
                    if (enable_kalman_filter) {
                        pos->x = filtered_vec[0];
                        pos->y = filtered_vec[1];
                        pos->z = filtered_vec[2];
                    } else {
                        pos->x = camera_position_vec[0];
                        pos->y = camera_position_vec[1];
                        pos->z = camera_position_vec[2];
                    }
                    pos->roll = aruco_result.roll_pitch_yaw[0];
                    pos->pitch = aruco_result.roll_pitch_yaw[1];
                    pos->yaw = aruco_result.roll_pitch_yaw[2];
                    pos_updated = true;
                }
            } else {
                // ── Blob tracker mode (original) ────────────────────
                std::vector<PoseResult> results = tracker.processFrame(im, current_time_sec, cameraMatrix, distCoeffs, marker_points, blob_area_threshold);

                for (const auto& b : tracker.current_frame_blobs_info) {
                    blob_entries.push_back({
                        {"x", b.x},
                        {"y", b.y},
                        {"id", b.id}
                    });
                }

                for (const auto& result : results) {
                    std::vector<double> camera_position_vec = matVec3ToVector(result.tvec);
                    std::vector<double> camera_orientation_vec = vec3dToVector(result.roll_pitch_yaw);
                    std::vector<double> marker_position_vec = matVec3ToVector(result.marker_tvec);
                    std::vector<double> marker_orientation_vec = vec3dToVector(result.marker_roll_pitch_yaw);

                    if (print_logs) {
                        cout << "ID: " << result.marker_id
                             << " Camera Position: " << result.tvec.t()
                             << " Camera RPY: ["
                             << camera_orientation_vec[0] << ", " << camera_orientation_vec[1] << ", " << camera_orientation_vec[2] << "]"
                             << " Marker Position: " << result.marker_tvec.t()
                             << " Marker RPY: ["
                             << marker_orientation_vec[0] << ", " << marker_orientation_vec[1] << ", " << marker_orientation_vec[2] << "]" << endl;
                    }

                    json pose_entry = {
                        {"marker_id", result.marker_id},
                        {"camera_position", camera_position_vec},
                        {"camera_orientation", camera_orientation_vec},
                        {"marker_position", marker_position_vec},
                        {"marker_orientation", marker_orientation_vec}};

                    std::vector<double> filtered_vec = marker_position_vec;
                    if (enable_kalman_filter) {
                        if (kalman_filters.find(result.marker_id) == kalman_filters.end()) {
                            kalman_filters.emplace(result.marker_id,
                                                   PositionKalmanFilter(kf_process_noise, kf_measurement_noise));
                        }
                        Eigen::Vector3d meas(marker_position_vec[0], marker_position_vec[1], marker_position_vec[2]);
                        Eigen::Vector3d filt = kalman_filters.at(result.marker_id).update(meas, current_time_sec);
                        filtered_vec = {filt[0], filt[1], filt[2]};
                        pose_entry["marker_position_filtered"] = filtered_vec;
                    }

                    current_frame_poses.push_back(pose_entry);

                    if (!pos_updated && (target_id == -1 || result.marker_id == target_id)) {
                        pos->valid = true;
                        if (enable_kalman_filter) {
                            pos->x = filtered_vec[0];
                            pos->y = filtered_vec[1];
                            pos->z = filtered_vec[2];
                        } else {
                            pos->x = marker_position_vec[0];
                            pos->y = marker_position_vec[1];
                            pos->z = marker_position_vec[2];
                        }
                        pos->roll = marker_orientation_vec[0];
                        pos->pitch = marker_orientation_vec[1];
                        pos->yaw = marker_orientation_vec[2];
                        pos_updated = true;
                    }
                }
            }

            if (!pos_updated) {
                pos->valid = false;
            }
            pos->timestamp = current_time_sec;

            json frame_data = {
                {"time", current_time_sec},
                {"frame_id", frameCount},
                {"poses", current_frame_poses}
            };
            if (!aruco_mode) {
                frame_data["blobs"] = blob_entries;
            }
            frames.push_back(frame_data);

            Mat stream_im = raw_stream ? raw_im : im;

            // Update streaming frame (only if streaming is enabled and frame is valid)
            if (enable_streaming && streamer && !stream_im.empty() && current_time_sec >= stream_next_time) {
                try {
                    streamer->updateFrame(stream_im);
                    stream_next_time = (stream_next_time == 0) ? current_time_sec + stream_interval : stream_next_time + stream_interval;
                    if (stream_next_time < current_time_sec)
                        stream_next_time = current_time_sec + stream_interval;
                } catch (const std::exception& e) {
                    cerr << "Streaming error: " << e.what() << endl;
                }
            }

            Mat preview_im = raw_preview ? raw_im : im;
            if (preview) {
                imshow("libcamera-demo", preview_im);
            }

            // Handle background saving for images and video
            bool do_save_image = false;
            bool do_save_video = false;
            string save_filename = "";

            if (save_frames && current_time_sec >= image_next_time) {
                do_save_image = true;
                // Only advance next_time by interval, or reset if it's falling way behind
                image_next_time = (image_next_time == 0) ? current_time_sec + image_interval : image_next_time + image_interval;
                if (image_next_time < current_time_sec)
                    image_next_time = current_time_sec + image_interval;

                string save_dir = save_frames_path.empty() ? log_dir : save_frames_path;
                save_filename = save_dir + "/frame_" + to_string(frameCount) + ".jpg";
            }

            Mat save_video_im = raw_save_video ? raw_im : im;
            if (save_video && current_time_sec >= video_next_time) {
                if (!saver.isVideoOpened() && !save_video_im.empty()) {
                    int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
                    if (saver.startVideo(video_filename, codec, video_fps, save_video_im.size(), save_video_im.channels() == 3)) {
                        video_start_time = current_time_sec;
                        // Initialize next time to now + interval
                        video_next_time = current_time_sec + video_interval;
                    } else {
                        cerr << "Could not open the output video file for write" << endl;
                        save_video = false;
                    }
                }

                if (saver.isVideoOpened()) {
                    do_save_video = true;
                    // Only advance next_time by interval, or reset if it's falling way behind
                    video_next_time = (video_next_time == 0) ? current_time_sec + video_interval : video_next_time + video_interval;
                    if (video_next_time < current_time_sec)
                        video_next_time = current_time_sec + video_interval;
                }
            }

            Mat save_frame_im = raw_save_frame ? raw_im : im;
            if (do_save_image && do_save_video) {
                if (raw_save_frame == raw_save_video) {
                    if (!save_frame_im.empty())
                        saver.push(save_frame_im, save_filename, true);
                } else {
                    if (!save_frame_im.empty())
                        saver.push(save_frame_im, save_filename, false);
                    if (!save_video_im.empty())
                        saver.push(save_video_im, "", true);
                }
            } else if (do_save_image) {
                if (!save_frame_im.empty())
                    saver.push(save_frame_im, save_filename, false);
            } else if (do_save_video) {
                if (!save_video_im.empty())
                    saver.push(save_video_im, "", true);
            }

            if (preview) {
                key = waitKey(1);
            } else {
                key = -1;
            }
            if (key == 'q') {
                break;
            } else if (key == 'f') {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeAuto);
                controls.set(controls::AfTrigger, 0);
                cam.set(controls);
            } else if (key == 'a' || key == 'A') {
                lens_position += focus_step;
            } else if (key == 'd' || key == 'D') {
                lens_position -= focus_step;
            }

            if (key == 'a' || key == 'A' || key == 'd' || key == 'D') {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeManual);
                controls.set(controls::LensPosition, lens_position);
                cam.set(controls);
            }

            frame_count++;
            frameCount++;
            if ((time(0) - start_time) >= 1) {
                if (print_logs) {
                    cout << frame_count << "fps" << endl;
                }
                frame_count = 0;
                start_time = time(0);
                elapsed_seconds += 1;
            }

            if (execution_time > 0 && elapsed_seconds >= execution_time) {
                break;
            }
            cam.returnFrameBuffer(frameData);

            // Pace the loop to maintain frame_rate without affecting camera exposure
            next_frame_time += frame_interval;
            std::this_thread::sleep_until(next_frame_time);
        }

        saver.stop();

        if (save_frames) {
            string save_dir = save_frames_path.empty() ? log_dir : save_frames_path;
            string zip_filename = save_dir + "/frames.zip";
            string zip_command = "zip -q -j -m " + zip_filename + " " + save_dir + "/*.jpg 2>/dev/null";
            cout << "Zipping frames to " << zip_filename << "..." << endl;
            int ret = system(zip_command.c_str());
            if (ret == 0) {
                cout << "Successfully zipped frames and removed originals." << endl;
                cout << "Zip file path: " << zip_filename << endl;
            } else {
                cerr << "Failed to zip frames or no frames found." << endl;
            }
        }

        destroyAllWindows();
        cam.stopCamera();

        string log_filename = json_path.empty() ? (log_dir + "/log.json") : json_path;
        json log;
        log["args"] = {
            {"print_logs", print_logs},
            {"preview", preview},
            {"distance", distance},
            {"execution_time", execution_time},
            {"save_rate", save_rate},
            {"save_frames", save_frames},
            {"save_frames_path", save_frames_path},
            {"save_video", save_video},
            {"video_fps", video_fps},
            {"video_path", video_path},
            {"config_file", config_file},
            {"json_path", json_path},
            {"raw_preview", raw_preview},
            {"raw_stream", raw_stream},
            {"raw_save_frame", raw_save_frame},
            {"raw_save_video", raw_save_video},
            {"aruco_mode", aruco_mode},
            {"enable_streaming", enable_streaming},
            {"stream_port", stream_port},
            {"stream_type", stream_type},
            {"stream_rate", stream_rate},
            {"contrast", contrast},
            {"brightness", brightness},
            {"exposure_time", exposure_time},
            {"frame_rate", frame_rate},
            {"encoder_frame_rate", encoder_frame_rate},
            {"cam_width", cam_width},
            {"cam_height", cam_height},
            {"blob_area_threshold", blob_area_threshold},
            {"payload_size", payload_size},
            {"target_id", target_id},
            {"tracking_threshold", tracking_threshold},
            {"sync_threshold", sync_threshold},
            {"static_markers_mode", static_markers_mode},
            {"enable_kalman_filter", enable_kalman_filter},
            {"kf_process_noise", kf_process_noise},
            {"kf_measurement_noise", kf_measurement_noise}
        };
        log["config"] = {{"distance", distance}};
        log["config"]["git_version"] = getGitCommitHash();
        log["config"]["aruco_mode"] = aruco_mode;
        if (video_start_time > 0) {
            log["config"]["video_start_time"] = video_start_time;
        }
        log["config"]["kalman_filter_enabled"] = enable_kalman_filter;
        if (enable_kalman_filter) {
            log["config"]["kf_process_noise"] = kf_process_noise;
            log["config"]["kf_measurement_noise"] = kf_measurement_noise;
        }
        log["frames"] = frames;

        std::ofstream file(log_filename);
        if (file.is_open()) {
            file << log.dump(4);
            file.close();
            std::cout << "Logs saved to " << log_filename << std::endl;
        } else {
            std::cerr << "Failed to write logs to file." << std::endl;
        }
    }

    // Clean up streaming
    if (streamer) {
        delete streamer;
    }

    // Clean up ArUco tracker
    if (aruco_tracker) {
        delete aruco_tracker;
    }

    cam.closeCamera();
    return 0;
}
