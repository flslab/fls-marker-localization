#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include "LibCamera.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>


using namespace cv;
using namespace std;


struct PoseResult {
    Mat img;
    Mat tvec;
    Mat rmat;
    Vec3d yaw_pitch_roll;
};

// Function to get a timestamped filename
std::string generateFilename() {
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    std::ostringstream filename;
    filename << "pose_logs_"
             << std::put_time(localTime, "%Y-%m-%d_%H-%M-%S")
             << ".txt";

    return filename.str();
}

Vec3d yawPitchRollDecomposition(const Mat& rmat) {
    double yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
    double pitch = atan2(-rmat.at<double>(2, 0),
                         sqrt(pow(rmat.at<double>(2, 1), 2) + pow(rmat.at<double>(2, 2), 2)));
    double roll = atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));
    return Vec3d(yaw, pitch, roll);
}


// Main function to process an image and compute pose
PoseResult processImage(const Mat& input, const Mat& cameraMatrix, const Mat& distCoeffs) {
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
    for (const auto& contour : contours) {
        Moments moments = cv::moments(contour);
        if (moments.m00 > 50) {
            int center_x = int(moments.m10 / moments.m00);
            int center_y = int(moments.m01 / moments.m00);
            circle(im, cv::Point(center_x, center_y), 10, Scalar(0, 0, 255), -1);
            image_points.push_back(Point2f(center_x, center_y));
        }
    }

    // Step 5: Validate image points
    if (image_points.size() != 4) {
//        cout << "Not enough points found!" << endl;
        return {im, Mat(), Mat(), Vec3d()};
    }

    // Step 6: Marker 3D points
    vector<Point3f> marker_points = {
        {-0.030, 0.006, 0.0},
        {-0.011, 0.023, 0.0},
        {0.024, 0.0, 0.0},
        {-0.007, -0.026, 0.0}
    };

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


int main(int argc, char **argv) {
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
	controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({ frame_time, frame_time }));
    // Adjust the brightness of the output images, in the range -1.0 to 1.0
//    controls_.set(controls::Brightness, 0.5);
    // Adjust the contrast of the output image, where 1.0 = normal contrast
//    controls_.set(controls::Contrast, 1.5);
    // Set the exposure time
    // controls_.set(controls::ExposureTime, 20000);
    cam.set(controls_);

    // Camera matrix and distortion coefficients (replace with actual calibration values)
    Mat cameraMatrix = (Mat_<double>(3, 3) << 836.84712717, 0.0, 643.78276191,
                                             0.0, 836.78674967, 361.98923042,
                                             0.0, 0.0, 1.0);
    Mat distCoeffs = (Mat_<double>(5, 1) << -0.05726028, 0.08830815, 0.00106169, 0.00274017, 0.05117629);

    std::ostringstream logStream;

    if (!ret) {
        bool flag;
        LibcameraOutData frameData;
        cam.startCamera();
        cam.VideoStream(&width, &height, &stride);
        while (true) {
            flag = cam.readFrame(&frameData);
            if (!flag)
                continue;
            // CV_8UC3 for color CV_8UC1 for grayscale image
            Mat im(height, width, CV_8UC3, frameData.imageData, stride);

            cv::Mat frame;
//            cv::cvtColor(im, frame, cv::COLOR_BGR2GRAY);
            // Mat frame(height, width, CV_8UC1, frameData.imageData, stride);

            // Process the image
            PoseResult result = processImage(im, cameraMatrix, distCoeffs);

            // Display results
            if (!result.tvec.empty()) {
//                cout << "Translation Vector: " << result.tvec.t() << endl;
//                cout << "Yaw, Pitch, Roll: " << result.yaw_pitch_roll << endl;

                logStream << "Frame " << frameCount << ":" << std::endl;
                logStream << "    Translation Vector: " << result.tvec.t() << std::endl;
                logStream << "    Yaw, Pitch, Roll: " << result.yaw_pitch_roll << std::endl;
            } else {
//                cout << "Failed to compute pose!" << endl;
            }


            imshow("libcamera-demo", result.img);
            key = waitKey(1);
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

            // To use the manual focus function, libcamera-dev needs to be updated to version 0.0.10 and above.
            if (key == 'a' || key == 'A' || key == 'd' || key == 'D') {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeManual);
				controls.set(controls::LensPosition, lens_position);
                cam.set(controls);
            }

            frame_count++;
            frameCount++;
            if ((time(0) - start_time) >= 1){
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
