#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include "LibCamera.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

//#pragma pack(1)

using namespace cv;
using namespace std;
using json = nlohmann::json;

struct PoseResult
{
    Mat img;
    Mat tvec;
    Mat rmat;
    Vec3d yaw_pitch_roll;
};

struct Position {
    bool valid;
    float x, y, z;
    float roll, pitch, yaw;
};

// Video streaming class
class VideoStreamer {
private:
    int socket_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len;
    bool is_running;
    std::thread streaming_thread;
    std::atomic<bool> new_frame_available;
    Mat current_frame;
    std::mutex frame_mutex;
    int stream_port;
    string stream_type;

public:
    VideoStreamer(int port = 8080, const string& type = "udp")
        : stream_port(port), stream_type(type), is_running(false), new_frame_available(false) {
        client_len = sizeof(client_addr);
    }

    ~VideoStreamer() {
        stop();
    }

    bool start() {
        if (stream_type == "udp") {
            return startUDPStreaming();
        } else if (stream_type == "http") {
            return startHTTPStreaming();
        }
        return false;
    }

    bool startUDPStreaming() {
        socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd < 0) {
            cerr << "Error creating UDP socket" << endl;
            return false;
        }

        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(stream_port);

        if (bind(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            cerr << "Error binding UDP socket to port " << stream_port << endl;
            close(socket_fd);
            return false;
        }

        is_running = true;
        streaming_thread = std::thread(&VideoStreamer::udpStreamingLoop, this);
        cout << "UDP streaming started on port " << stream_port << endl;
        return true;
    }

    bool startHTTPStreaming() {
        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd < 0) {
            cerr << "Error creating HTTP socket" << endl;
            return false;
        }

        int opt = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(stream_port);

        if (bind(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            cerr << "Error binding HTTP socket to port " << stream_port << endl;
            close(socket_fd);
            return false;
        }

        if (listen(socket_fd, 5) < 0) {
            cerr << "Error listening on HTTP socket" << endl;
            close(socket_fd);
            return false;
        }

        is_running = true;
        streaming_thread = std::thread(&VideoStreamer::httpStreamingLoop, this);
        cout << "HTTP streaming started on port " << stream_port << endl;
        cout << "Open http://localhost:" << stream_port << "/stream in your browser" << endl;
        return true;
    }

    void updateFrame(const Mat& frame) {
        if (frame.empty()) return;

        std::lock_guard<std::mutex> lock(frame_mutex);

        // Ensure proper memory alignment and continuous memory layout
        if (frame.isContinuous()) {
            current_frame = frame.clone();
        } else {
            // Create a continuous copy if the frame is not continuous
            Mat temp_frame;
            frame.copyTo(temp_frame);
            current_frame = temp_frame;
        }

        new_frame_available = true;
    }

    void stop() {
        is_running = false;
        if (streaming_thread.joinable()) {
            streaming_thread.join();
        }
        if (socket_fd >= 0) {
            close(socket_fd);
        }
    }

private:
    void udpStreamingLoop() {
        vector<uchar> buffer;
        vector<int> encode_params = {IMWRITE_JPEG_QUALITY, 60}; // Lower quality for stability

        // Wait for initial client connection with timeout
        char dummy_buffer[1];
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        cout << "Waiting for UDP client connection..." << endl;
        if (recvfrom(socket_fd, dummy_buffer, 1, 0, (struct sockaddr*)&client_addr, &client_len) < 0) {
            cout << "No UDP client connected, continuing without streaming..." << endl;
            return;
        }
        cout << "UDP client connected from " << inet_ntoa(client_addr.sin_addr) << endl;

        while (is_running) {
            if (new_frame_available.load()) {
                Mat frame_to_send;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    if (!current_frame.empty() && current_frame.isContinuous()) {
                        frame_to_send = current_frame.clone();
                    }
                    new_frame_available = false;
                }

                if (!frame_to_send.empty()) {
                    // Resize frame for network efficiency
                    Mat resized_frame;
                    if (frame_to_send.cols > 320) {
                        resize(frame_to_send, resized_frame, cv::Size(320, 240));
                    } else {
                        resized_frame = frame_to_send;
                    }

                    // Encode frame as JPEG with error checking
                    buffer.clear();
                    try {
                        if (imencode(".jpg", resized_frame, buffer, encode_params) && !buffer.empty()) {
                            // Limit max frame size
                            if (buffer.size() < 65536) { // 64KB limit
                                // Send frame size first
                                uint32_t frame_size = htonl(buffer.size());
                                if (sendto(socket_fd, &frame_size, sizeof(frame_size), 0,
                                          (struct sockaddr*)&client_addr, client_len) < 0) {
                                    break; // Client disconnected
                                }

                                // Send frame data in smaller chunks
                                const size_t chunk_size = 512; // Smaller chunks for stability
                                size_t bytes_sent = 0;
                                while (bytes_sent < buffer.size() && is_running) {
                                    size_t remaining = buffer.size() - bytes_sent;
                                    size_t to_send = min(chunk_size, remaining);

                                    if (sendto(socket_fd, buffer.data() + bytes_sent, to_send, 0,
                                              (struct sockaddr*)&client_addr, client_len) < 0) {
                                        goto udp_loop_end; // Break out of nested loops
                                    }
                                    bytes_sent += to_send;
                                }
                            }
                        }
                    } catch (const cv::Exception& e) {
                        cerr << "OpenCV encoding error: " << e.what() << endl;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // ~20 FPS for stability
        }

        udp_loop_end:
        cout << "UDP streaming ended" << endl;
    }

    void httpStreamingLoop() {
        while (is_running) {
            struct timeval timeout;
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

            int client_socket = accept(socket_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_socket < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue; // Timeout, try again
                }
                break; // Real error
            }

            cout << "HTTP client connected from " << inet_ntoa(client_addr.sin_addr) << endl;

            // Send HTTP headers for MJPEG stream
            string headers =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Connection: keep-alive\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n\r\n";

            if (send(client_socket, headers.c_str(), headers.length(), MSG_NOSIGNAL) < 0) {
                close(client_socket);
                continue;
            }

            vector<uchar> buffer;
            vector<int> encode_params = {IMWRITE_JPEG_QUALITY, 60}; // Lower quality for stability

            while (is_running) {
                if (new_frame_available.load()) {
                    Mat frame_to_send;
                    {
                        std::lock_guard<std::mutex> lock(frame_mutex);
                        if (!current_frame.empty() && current_frame.isContinuous()) {
                            frame_to_send = current_frame.clone();
                        }
                        new_frame_available = false;
                    }

                    if (!frame_to_send.empty()) {
                        // Resize frame for network efficiency
                        Mat resized_frame;
                        if (frame_to_send.cols > 320) {
                            resize(frame_to_send, resized_frame, cv::Size(320, 240));
                        } else {
                            resized_frame = frame_to_send;
                        }

                        buffer.clear();
                        try {
                            if (imencode(".jpg", resized_frame, buffer, encode_params) && !buffer.empty()) {
                                // Limit max frame size
                                if (buffer.size() < 65536) { // 64KB limit
                                    string frame_header =
                                        "--frame\r\n"
                                        "Content-Type: image/jpeg\r\n"
                                        "Content-Length: " + to_string(buffer.size()) + "\r\n\r\n";

                                    if (send(client_socket, frame_header.c_str(), frame_header.length(), MSG_NOSIGNAL) < 0 ||
                                        send(client_socket, buffer.data(), buffer.size(), MSG_NOSIGNAL) < 0 ||
                                        send(client_socket, "\r\n", 2, MSG_NOSIGNAL) < 0) {
                                        break; // Client disconnected
                                    }
                                }
                            }
                        } catch (const cv::Exception& e) {
                            cerr << "OpenCV encoding error: " << e.what() << endl;
                            break;
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50)); // ~20 FPS for stability
            }

            close(client_socket);
            cout << "HTTP client disconnected" << endl;
        }
    }
};

// Function to get a timestamped filename
std::string generateLogName()
{
    std::time_t now = std::time(nullptr);
    std::tm *localTime = std::localtime(&now);

    std::ostringstream filename;
    filename << "logs/"
             << std::put_time(localTime, "%H_%M_%S_%m_%d_%Y");

    return filename.str();
}

Vec3d yawPitchRollDecomposition(const Mat &rmat)
{
    double yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
    double pitch = atan2(-rmat.at<double>(2, 0),
                         sqrt(pow(rmat.at<double>(2, 1), 2) + pow(rmat.at<double>(2, 2), 2)));
    double roll = atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));
    return Vec3d(yaw, pitch, roll);
}

// Function to calculate the centroid
Point2f findCentroid(const vector<Point2f>& points) {
    float cx = 0, cy = 0;
    for (const auto& p : points) {
        cx += p.x;
        cy += p.y;
    }
    return {cx / points.size(), cy / points.size()};
}

// Comparator to sort points clockwise around the centroid
bool clockwiseComparator(const Point2f& a, const Point2f& b, const Point2f& center) {
    // Calculate angles relative to the center
    double angleA = atan2(a.y - center.y, a.x - center.x);
    double angleB = atan2(b.y - center.y, b.x - center.x);
    return angleA < angleB; // Clockwise order
}

// Function to sort points in clockwise order
void sortClockwise(vector<Point2f>& points) {
    // Find the centroid
    Point2f center = findCentroid(points);

    // Sort using the comparator
    sort(points.begin(), points.end(), [&](const Point2f& a, const Point2f& b) {
        return clockwiseComparator(a, b, center);
    });
}

// Main function to process an image and compute pose
PoseResult processImage(const Mat &input, const Mat &cameraMatrix, const Mat &distCoeffs, const vector<Point3f> &marker_points)
{
    Mat im = input.clone();

    // Apply Gaussian Blur
    GaussianBlur(im, im, cv::Size(9, 9), 0);

    // Convert to grayscale and threshold
    Mat grey;
    cvtColor(im, grey, COLOR_BGR2GRAY);
    threshold(grey, grey, 255 * 0.8, 255, THRESH_BINARY);

    // Find contours
    vector<vector<cv::Point>> contours;
    findContours(grey, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(im, contours, -1, Scalar(255, 0, 0), 2);

    // Find image points from contours
    vector<Point2f> image_points;
    for (const auto &contour : contours)
    {
        Moments moments = cv::moments(contour);
        if (moments.m00 > 9)
        {
            int center_x = int(moments.m10 / moments.m00);
            int center_y = int(moments.m01 / moments.m00);
            circle(im, cv::Point(center_x, center_y), 10, Scalar(0, 0, 255), 1);
            image_points.push_back(Point2f(center_x, center_y));
        }
    }

    // Validate image points
    if (image_points.size() != 4)
    {
        return {im, Mat(), Mat(), Vec3d()};
    }

    // Use provided marker points
    if (marker_points.size() != 4)
    {
        return {im, Mat(), Mat(), Vec3d()};
    }

    sortClockwise(image_points);

    // SolvePnP
    Mat rvec, tvec;
    solvePnP(marker_points, image_points, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_AP3P);

    // Convert rotation vector to matrix
    Mat rmat;
    Rodrigues(rvec, rmat);

    // Extract pose and orientation
    Vec3d yaw_pitch_roll = yawPitchRollDecomposition(rmat);

    return {im, tvec, rmat, yaw_pitch_roll};
}

bool readConfigFile(const string &filename, Mat &cameraMatrix, Mat &distCoeffs, vector<Point3f> &marker_points)
{
    try
    {
        // Read JSON file
        std::ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Failed to open config file: " << filename << endl;
            return false;
        }
        json config = json::parse(file);

        // Read camera matrix
        cameraMatrix = Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cameraMatrix.at<double>(i, j) = config["/camera_matrix"_json_pointer][i][j];
            }
        }

        // Read distortion coefficients
        vector<double> dist = config["dist_coeffs"];
        distCoeffs = Mat(dist, true);

        // Read marker points
        marker_points.clear();
        for (const auto &point : config["marker_points"])
        {
            marker_points.push_back(Point3f(point[0], point[1], point[2]));
        }

        return true;
    }
    catch (const exception &e)
    {
        cerr << "Error reading config file: " << e.what() << endl;
        return false;
    }
}

bool createDirectory(const string &dir) {
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        // Directory does not exist, create it
        return mkdir(dir.c_str(), 0777) == 0;
    }
    return S_ISDIR(info.st_mode); // Check if it's a directory
}

int main(int argc, char **argv)
{
    bool print_logs = false;
    bool preview = false;
    double distance = -1.0;
    int execution_time = 0;
    int save_rate = 1;
    bool save_frames = false;
    string config_file = "camera_config.json";

    // Streaming parameters
    bool enable_streaming = false;
    int stream_port = 8080;
    string stream_type = "http"; // "http" or "udp"
    int stream_rate = 10;

    double contrast = -2.0;
    double brightness = -2.0;
    int exposure_time = -2;
    int frame_rate = 120;

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
            } catch (const invalid_argument &e) {
                cerr << "Invalid value for distance. Must be a positive number." << endl;
                return -1;
            }
        } else if ((arg == "--time" || arg == "-t") && i + 1 < argc) {
            try {
                execution_time = stoi(argv[++i]);
                if (execution_time <= 0) {
                    throw invalid_argument("Time must be positive");
                }
            } catch (const invalid_argument &e) {
                cerr << "Invalid value for time. Must be a positive number." << endl;
                return -1;
            }
        } else if ((arg == "--save-frames" || arg == "-s")) {
            save_frames = true;
        } else if ((arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        } else if ((arg == "--save-rate") && i + 1 < argc) {
            save_rate = stoi(argv[++i]);
        } else if ((arg == "--contrast") && i + 1 < argc) {
            contrast = stod(argv[++i]);
        } else if ((arg == "--brightness") && i + 1 < argc) {
            brightness = stod(argv[++i]);
        } else if ((arg == "--exposure") && i + 1 < argc) {
            exposure_time = stoi(argv[++i]);
        } else if ((arg == "--fps") && i + 1 < argc) {
            frame_rate = stoi(argv[++i]);
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
            stream_rate = stoi(argv[++i]);
        }
    }

    string log_dir = generateLogName();
    if (!createDirectory(log_dir)) {
        cerr << "Error: Unable to create directory " << log_dir << endl;
        return -1;
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
        cout << "Streamin at " << frame_rate / stream_rate << " fps" << endl;
    }

    if (save_frames) {
        cout << "Saving frames at " << frame_rate / save_rate << " fps" << endl;
    }

    time_t start_time = time(0);
    int frame_count = 0;
    int frameCount = 0;
    int elapsed_seconds = 0;
    float lens_position = 100;
    float focus_step = 50;
    LibCamera cam;
    uint32_t width = 640;
    uint32_t height = 400;
    uint32_t stride;
    char key;
    int window_width = 640;
    int window_height = 400;

    if (preview) {
        if (width > window_width)
        {
            cv::namedWindow("libcamera-demo", cv::WINDOW_NORMAL);
            cv::resizeWindow("libcamera-demo", window_width, window_height);
        }
    }

    int ret = cam.initCamera();
    cam.configureStill(width, height, formats::R8, 1, 0);
    ControlList controls_;
    int64_t frame_time = 1000000 / frame_rate;
    controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_time, frame_time}));

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

    if (!readConfigFile(config_file, cameraMatrix, distCoeffs, marker_points))
    {
        cerr << "Failed to read camera configuration" << endl;
        return -1;
    }

    if (!ret)
    {
        bool flag;
        LibcameraOutData frameData;
        cam.startCamera();
        cam.VideoStream(&width, &height, &stride);
        std::vector<json> frames;

        const char* shm_name = "/pos_shared_mem";
        int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
        ftruncate(shm_fd, sizeof(Position));
        Position* pos = (Position*)mmap(0, sizeof(Position), PROT_WRITE, MAP_SHARED, shm_fd, 0);

        while (true)
        {
            flag = cam.readFrame(&frameData);
            if (!flag)
                continue;

            // Create a properly aligned and continuous Mat from camera data
            Mat raw_frame(height, width, CV_16UC1, frameData.imageData, stride);
            raw_frame.convertTo(raw_frame, CV_8UC3, 255.0 / 1023.0);
            Mat im;

            // Ensure the frame is continuous and properly aligned
            if (raw_frame.isContinuous() && stride == width * 3) {
                im = raw_frame.clone(); // Safe copy
            } else {
                // Create a properly aligned copy
                raw_frame.copyTo(im);
            }

            // Process the image
            PoseResult result = processImage(im, cameraMatrix, distCoeffs, marker_points);

            if (!result.tvec.empty()) {
                if (print_logs) {
                    cout << "Translation Vector: " << result.tvec.t() << "Yaw, Pitch, Roll: " << result.yaw_pitch_roll << endl;
                }

                std::vector<double> tvec_vec = {result.tvec.at<double>(0, 0),
                                        result.tvec.at<double>(1, 0),
                                        result.tvec.at<double>(2, 0)};

                auto now = std::chrono::system_clock::now();
                auto duration = now.time_since_epoch();
                auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

                frames.push_back({
                    {"time", milliseconds},
                    {"frame_id", frameCount},
                    {"tvec", tvec_vec},
                    {"yaw_pitch_roll", {result.yaw_pitch_roll[0], result.yaw_pitch_roll[1], result.yaw_pitch_roll[2]}}
                });

                pos->valid = true;
                pos->x = tvec_vec[0];
                pos->y = tvec_vec[1];
                pos->z = tvec_vec[2];
                pos->roll = result.yaw_pitch_roll[2];
                pos->pitch = result.yaw_pitch_roll[1];
                pos->yaw = result.yaw_pitch_roll[0];
            }
            else {
                pos->valid = false;
            }

            // Update streaming frame (only if streaming is enabled and frame is valid)
            if (enable_streaming && frameCount % stream_rate == 0 && streamer && !result.img.empty()) {
                try {
                    streamer->updateFrame(result.img);
                } catch (const std::exception& e) {
                    cerr << "Streaming error: " << e.what() << endl;
                }
            }

            if (preview) {
                imshow("libcamera-demo", result.img);
            }

            // Save frames if enabled
            if (save_frames && frameCount % save_rate == 0) {
                string filename = log_dir + "/frame_" + to_string(frameCount) + ".png";
                imwrite(filename, result.img);
            }

            key = waitKey(1);
            if (key == 'q') {
                break;
            }
            else if (key == 'f') {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeAuto);
                controls.set(controls::AfTrigger, 0);
                cam.set(controls);
            }
            else if (key == 'a' || key == 'A') {
                lens_position += focus_step;
            }
            else if (key == 'd' || key == 'D') {
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
        }

        destroyAllWindows();
        cam.stopCamera();

        string log_filename = log_dir + "/log.json";
        json log;
        log["config"] = {{"distance", distance}};
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

    cam.closeCamera();
    return 0;
}