#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include "LibCamera.h"


using namespace cv;
using namespace std;


struct PoseResult {
    Mat img;
    Mat tvec;
    Mat rmat;
    Vec3d yaw_pitch_roll;
};


// Main function to process an image and compute pose
PoseResult processImage(const Mat& input, const Mat& cameraMatrix, const Mat& distCoeffs) {
    Mat im = input.clone();

    // Step 1: Apply Gaussian Blur
    GaussianBlur(im, im, cv::Size(9, 9), 0);

    // Step 2: Convert to grayscale and threshold
    Mat grey;
    cvtColor(im, grey, COLOR_BGR2GRAY);
    threshold(grey, grey, 255 * 0.75, 255, THRESH_BINARY);

    // Step 3: Find contours
    vector<vector<Point>> contours;
    findContours(grey, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(im, contours, -1, Scalar(255, 0, 0), 2);

    // Step 4: Find image points from contours
    vector<Point2f> image_points;
    for (const auto& contour : contours) {
        Moments moments = cv::moments(contour);
        if (moments.m00 > 100) {
            int center_x = int(moments.m10 / moments.m00);
            int center_y = int(moments.m01 / moments.m00);
            circle(im, Point(center_x, center_y), 10, Scalar(0, 0, 255), -1);
            image_points.push_back(Point2f(center_x, center_y));
        }
    }

    // Step 5: Validate image points
    if (image_points.size() < 4) {
        cout << "Not enough points found!" << endl;
        return {im, Mat(), Mat(), Vec3d()};
    }

    // Step 6: Marker 3D points
    vector<Point3f> marker_points = {
        {0.0, 0.0, 0.0},
        {0.0464, -0.002, 0.0},
        {0.0444, -0.0197, 0.0},
        {0.0094, -0.0291, 0.0}
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
    int64_t frame_time = 1000000 / 30;
    // Set frame rate
	controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({ frame_time, frame_time }));
    // Adjust the brightness of the output images, in the range -1.0 to 1.0
    controls_.set(controls::Brightness, 0.5);
    // Adjust the contrast of the output image, where 1.0 = normal contrast
    controls_.set(controls::Contrast, 1.5);
    // Set the exposure time
    // controls_.set(controls::ExposureTime, 20000);
    cam.set(controls_);

    // Camera matrix and distortion coefficients (replace with actual calibration values)
    Mat cameraMatrix = (Mat_<double>(3, 3) << 800, 0, 640,
                                             0, 800, 360,
                                             0, 0, 1);
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);

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
            cv::cvtColor(im, frame, cv::COLOR_BGR2GRAY);
            // Mat frame(height, width, CV_8UC1, frameData.imageData, stride);

            // Process the image
            PoseResult result = processImage(frame, cameraMatrix, distCoeffs);

            // Display results
            if (!result.tvec.empty()) {
                cout << "Translation Vector: " << result.tvec.t() << endl;
                cout << "Yaw, Pitch, Roll: " << result.yaw_pitch_roll << endl;
            } else {
                cout << "Failed to compute pose!" << endl;
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
            if ((time(0) - start_time) >= 1){
                printf("fps: %d\n", frame_count);
                frame_count = 0;
                start_time = time(0);
            }
            cam.returnFrameBuffer(frameData);
        }
        destroyAllWindows();
        cam.stopCamera();
    }
    cam.closeCamera();
    return 0;
}
