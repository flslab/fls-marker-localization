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
#include <sys/stat.h>  // For stat and mkdir
#include <sys/types.h> // For mode_t
#include <unistd.h>    // For access function
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <chrono>
#pragma pack(1)

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
    float qx, qy, qz, qw;
};

// Function to get a timestamped filename
std::string generateFilename()
{
    std::time_t now = std::time(nullptr);
    std::tm *localTime = std::localtime(&now);

    std::ostringstream filename;
    filename << "logs/pose_logs_"
             << std::put_time(localTime, "%Y-%m-%d_%H-%M-%S")
             << ".json";

    return filename.str();
}

// Eigen::Matrix3d cvMatToEigen(const cv::Mat &mat)
// {
//     Eigen::Matrix3d eigen_mat;
//     for (int i = 0; i < 3; i++)
//     {
//         for (int j = 0; j < 3; j++)
//         {
//             eigen_mat(i, j) = mat.at<double>(i, j);
//         }
//     }
//     return eigen_mat;
// }

// Vec3d yawPitchRollDecomposition(const Mat &rmat)
// {
//     // Convert OpenCV Mat to Eigen Matrix
//     Eigen::Matrix3d R = cvMatToEigen(rmat);

//     // Get euler angles in XYZ order (roll, pitch, yaw)
//     Eigen::Vector3d euler_angles = R.eulerAngles(0, 1, 2);

//     // Convert to degrees
//     const double rad2deg = 180.0 / M_PI;
//     euler_angles *= rad2deg;

//     // Return as OpenCV Vec3d (yaw, pitch, roll)
//     // Note: We reorder from XYZ (roll, pitch, yaw) to ZYX (yaw, pitch, roll)
//     return Vec3d(euler_angles[2], euler_angles[1], euler_angles[0]);
// }

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

    // Step 1: Apply Gaussian Blur
    GaussianBlur(im, im, cv::Size(9, 9), 0);

    // Step 2: Convert to grayscale and threshold
    Mat grey;
    cvtColor(im, grey, COLOR_BGR2GRAY);
    threshold(grey, grey, 255 * 0.8, 255, THRESH_BINARY);

    // Step 2.5: Morphological opening to remove noise
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(grey, grey, MORPH_OPEN, kernel);

    // Step 3: Find contours
    vector<vector<cv::Point>> contours;
    findContours(grey, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // Step 4: Filter contours
    vector<Point2f> image_points;
    for (const auto &contour : contours)
    {
        double area = contourArea(contour);
        if (area < 100) continue; // Skip small blobs

        // Circularity = 4π × Area / Perimeter²
        double perimeter = arcLength(contour, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity < 0.75) continue; // Skip non-circular shapes

        Moments moments = cv::moments(contour);
        if (moments.m00 == 0) continue;

        int center_x = int(moments.m10 / moments.m00);
        int center_y = int(moments.m01 / moments.m00);
        circle(im, cv::Point(center_x, center_y), 10, Scalar(0, 0, 255), -1);
        image_points.push_back(Point2f(center_x, center_y));
    }

    // Optional: Draw remaining contours
    drawContours(im, contours, -1, Scalar(255, 0, 0), 2);

    // Step 5: Validate image points
    if (image_points.size() != 4)
    {
        return {im, Mat(), Mat(), Vec3d()};
    }

    // Step 6: Use provided marker points
    if (marker_points.size() != 4)
    {
        return {im, Mat(), Mat(), Vec3d()};
    }

    sortClockwise(image_points);

    // Step 7: SolvePnP
    Mat rvec, tvec;
    solvePnP(marker_points, image_points, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_AP3P);

    // Step 8: Convert rotation vector to matrix
    Mat rmat;
    Rodrigues(rvec, rmat);

    // Step 9: Extract pose and orientation
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
    double distance = -1.0;  // Default invalid value for distance
    int execution_time = 0; // Default to unlimited execution
    string save_dir = "";
    string config_file = "camera_config.json";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "--verbose" || arg == "-v") {
            print_logs = true;
        } else if (arg == "--preview" || arg == "-p") {
            preview = true;
        } else if ((arg == "--distance" || arg == "-d") && i + 1 < argc) {
            try {
                distance = stod(argv[++i]);  // Convert the next argument to a double
                if (distance <= 0) {
                    throw invalid_argument("Distance must be positive");
                }
            } catch (const invalid_argument &e) {
                cerr << "Invalid value for distance. Must be a positive number." << endl;
                return -1;
            }
        } else if ((arg == "--time" || arg == "-t") && i + 1 < argc) {
            try {
                execution_time = stoi(argv[++i]);  // Convert the next argument to an integer
                if (execution_time <= 0) {
                    throw invalid_argument("Time must be positive");
                }
            } catch (const invalid_argument &e) {
                cerr << "Invalid value for time. Must be a positive number." << endl;
                return -1;
            }
        } else if ((arg == "--save_frames" || arg == "-s") && i + 1 < argc) {
            save_dir = argv[++i];
            if (!createDirectory(save_dir)) {
                cerr << "Error: Unable to create directory " << save_dir << endl;
                return -1;
            }
        } else if ((arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        }
    }

    time_t start_time = time(0);
    int frame_count = 0;
    int frameCount = 0;
    int elapsed_seconds = 0;
    float lens_position = 100;
    float focus_step = 50;
    LibCamera cam;
    uint32_t width = 640;
    uint32_t height = 480;
    uint32_t stride;
    char key;
    int window_width = 640;
    int window_height = 480;

    if (preview) {
        if (width > window_width)
        {
            cv::namedWindow("libcamera-demo", cv::WINDOW_NORMAL);
            cv::resizeWindow("libcamera-demo", window_width, window_height);
        }
    }

    int ret = cam.initCamera();
    cam.configureStill(width, height, formats::RGB888, 1, 0);
    ControlList controls_;
    // 30 fps
    int64_t frame_time = 1000000 / 120;
    // Set frame rate
    controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_time, frame_time}));
    // Adjust the brightness of the output images, in the range -1.0 to 1.0
    //    controls_.set(controls::Brightness, 0.5);
    // Adjust the contrast of the output image, where 1.0 = normal contrast
    //    controls_.set(controls::Contrast, 1.5);
    // Set the exposure time
    // controls_.set(controls::ExposureTime, 20000);
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
            // CV_8UC3 for color CV_8UC1 for grayscale image
            Mat im(height, width, CV_8UC3, frameData.imageData, stride);

            cv::Mat frame;
            //            cv::cvtColor(im, frame, cv::COLOR_BGR2GRAY);
            // Mat frame(height, width, CV_8UC1, frameData.imageData, stride);

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

                // Calculate the duration since the epoch
                auto duration = now.time_since_epoch();

                // Convert the duration to milliseconds
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
                pos->qx = 0.0;
                pos->qy = 0.0;
                pos->qz = 0.0;
                pos->qw = 0.0;
            }
            else {
                //                cout << "Failed to compute pose!" << endl;
                pos->valid = false;
            }

            if (preview) {
                imshow("libcamera-demo", result.img);
            }
            // Save frames if enabled
            if (!save_dir.empty()) {
                string filename = save_dir + "/frame_" + to_string(frameCount) + ".png";
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

            // To use the manual focus function, libcamera-dev needs to be updated to version 0.0.10 and above.
            if (key == 'a' || key == 'A' || key == 'd' || key == 'D') {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeManual);
                controls.set(controls::LensPosition, lens_position);
                cam.set(controls);
            }

            frame_count++;
            frameCount++;
            if ((time(0) - start_time) >= 1) {
                printf("fps: %d\n", frame_count);
                frame_count = 0;
                start_time = time(0);
                elapsed_seconds += 1;
            }

            if (elapsed_seconds >= execution_time) {
              break;
            }
            cam.returnFrameBuffer(frameData);
        }
        destroyAllWindows();
        cam.stopCamera();

        std::string filename = generateFilename();


        json log;
        log["config"] = {{"distance", distance}};
        log["frames"] = frames;

        // Write to JSON file
        std::ofstream file(filename);
        if (file.is_open()) {
            file << log.dump(4);  // Pretty-print JSON with 4-space indentation
            file.close();
            std::cout << "Logs saved to " << filename << std::endl;
        } else {
            std::cerr << "Failed to write logs to file." << std::endl;
        }

    }
    cam.closeCamera();
    return 0;
}
