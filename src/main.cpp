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

// Function to get a timestamped filename
std::string generateFilename()
{
    std::time_t now = std::time(nullptr);
    std::tm *localTime = std::localtime(&now);

    std::ostringstream filename;
    filename << "pose_logs_"
             << std::put_time(localTime, "%Y-%m-%d_%H-%M-%S")
             << ".txt";

    return filename.str();
}

Eigen::Matrix3d cvMatToEigen(const cv::Mat &mat)
{
    Eigen::Matrix3d eigen_mat;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            eigen_mat(i, j) = mat.at<double>(i, j);
        }
    }
    return eigen_mat;
}

Vec3d yawPitchRollDecomposition(const Mat &rmat)
{
    // Convert OpenCV Mat to Eigen Matrix
    Eigen::Matrix3d R = cvMatToEigen(rmat);

    // Get euler angles in XYZ order (roll, pitch, yaw)
    Eigen::Vector3d euler_angles = R.eulerAngles(0, 1, 2);

    // Convert to degrees
    const double rad2deg = 180.0 / M_PI;
    euler_angles *= rad2deg;

    // Return as OpenCV Vec3d (yaw, pitch, roll)
    // Note: We reorder from XYZ (roll, pitch, yaw) to ZYX (yaw, pitch, roll)
    return Vec3d(euler_angles[2], euler_angles[1], euler_angles[0]);
}

// Vec3d yawPitchRollDecomposition(const Mat &rmat)
// {
//     double yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
//     double pitch = atan2(-rmat.at<double>(2, 0),
//                          sqrt(pow(rmat.at<double>(2, 1), 2) + pow(rmat.at<double>(2, 2), 2)));
//     double roll = atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));
//     return Vec3d(yaw, pitch, roll);
// }

// Main function to process an image and compute pose
PoseResult processImage(const Mat &input, const Mat &cameraMatrix, const Mat &distCoeffs, const vector<Point3f> &marker_points)
{
    Mat im = input.clone();

    // Step 1: Apply Gaussian Blur
    GaussianBlur(im, im, cv::Size(9, 9), 0);

    // Step 2: Convert to grayscale and threshold
    Mat grey;
    cvtColor(im, grey, COLOR_BGR2GRAY);
    threshold(grey, grey, 255 * 0.65, 255, THRESH_BINARY);

    // Step 3: Find contours
    vector<vector<cv::Point>> contours;
    findContours(grey, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(im, contours, -1, Scalar(255, 0, 0), 2);

    // Step 4: Find image points from contours
    vector<Point2f> image_points;
    for (const auto &contour : contours)
    {
        Moments moments = cv::moments(contour);
        if (moments.m00 > 50)
        {
            int center_x = int(moments.m10 / moments.m00);
            int center_y = int(moments.m01 / moments.m00);
            circle(im, cv::Point(center_x, center_y), 10, Scalar(0, 0, 255), -1);
            image_points.push_back(Point2f(center_x, center_y));
        }
    }

    // Step 5: Validate image points
    if (image_points.size() != 4)
    {
        //        cout << "Not enough points found!" << endl;
        return {im, Mat(), Mat(), Vec3d()};
    }

    // Step 6: Use provided marker points
    if (marker_points.size() != 4)
    {
        return {im, Mat(), Mat(), Vec3d()};
    }

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

int main(int argc, char **argv)
{
    // if (argc != 2) {
    //     std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
    //     return -1;
    // }
    // Load input image
    // cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    // if (image.empty()) {
    //     std::cerr << "Could not load image: " << argv[1] << std::endl;
    //     return -1;
    // }
    //
    // // Convert grayscale to BGR for displaying color circles and ellipses
    // cv::Mat displayImage;
    // cv::cvtColor(image, displayImage, cv::COLOR_GRAY2BGR);

    time_t start_time = time(0);
    int frame_count = 0;
    int frameCount = 0;
    float lens_position = 100;
    float focus_step = 50;
    LibCamera cam;
    uint32_t width = 640;
    uint32_t height = 480;
    uint32_t stride;
    char key;
    int window_width = 640;
    int window_height = 480;

    if (width > window_width)
    {
        cv::namedWindow("libcamera-demo", cv::WINDOW_NORMAL);
        cv::resizeWindow("libcamera-demo", window_width, window_height);
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

    if (!readConfigFile("camera_config.json", cameraMatrix, distCoeffs, marker_points))
    {
        cerr << "Failed to read camera configuration" << endl;
        return -1;
    }

    std::ostringstream logStream;

    if (!ret)
    {
        bool flag;
        LibcameraOutData frameData;
        cam.startCamera();
        cam.VideoStream(&width, &height, &stride);
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

            // Display results
            if (!result.tvec.empty())
            {
                //                cout << "Translation Vector: " << result.tvec.t() << endl;
                //                cout << "Yaw, Pitch, Roll: " << result.yaw_pitch_roll << endl;

                logStream << "Frame " << frameCount << ":" << std::endl;
                logStream << "    Translation Vector: " << result.tvec.t() << std::endl;
                logStream << "    Yaw, Pitch, Roll: " << result.yaw_pitch_roll << std::endl;
            }
            else
            {
                //                cout << "Failed to compute pose!" << endl;
            }

            imshow("libcamera-demo", result.img);
            key = waitKey(1);
            if (key == 'q')
            {
                break;
            }
            else if (key == 'f')
            {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeAuto);
                controls.set(controls::AfTrigger, 0);
                cam.set(controls);
            }
            else if (key == 'a' || key == 'A')
            {
                lens_position += focus_step;
            }
            else if (key == 'd' || key == 'D')
            {
                lens_position -= focus_step;
            }

            // To use the manual focus function, libcamera-dev needs to be updated to version 0.0.10 and above.
            if (key == 'a' || key == 'A' || key == 'd' || key == 'D')
            {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeManual);
                controls.set(controls::LensPosition, lens_position);
                cam.set(controls);
            }

            frame_count++;
            frameCount++;
            if ((time(0) - start_time) >= 1)
            {
                printf("fps: %d\n", frame_count);
                frame_count = 0;
                start_time = time(0);
            }
            cam.returnFrameBuffer(frameData);
        }
        destroyAllWindows();
        cam.stopCamera();

        std::string filename = generateFilename();
        std::ofstream logFile(filename);
        logFile << logStream.str();
        logFile.close();

        std::cout << "All logs saved to: " << filename << std::endl;
    }
    cam.closeCamera();
    return 0;
}
