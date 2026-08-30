#pragma once
/**
 * ArUco-based camera localization.
 *
 * Given a set of ArUco markers whose world poses are known, this class
 * detects markers in each frame, solves PnP per marker to obtain the
 * camera pose relative to each marker, transforms each result into the
 * world frame, and fuses the estimates via a weighted average (weights
 * are the inverse of the reprojection error).
 *
 * Supports both:
 *   - OpenCV 4.7+  (cv::aruco::ArucoDetector, objdetect module)
 *   - OpenCV 4.0–4.6 / contrib  (cv::aruco::detectMarkers, aruco module)
 */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/version.hpp>

#include "pose_estimator.h"
#include "pose_math.h"

// ─── ArUco API version detection ────────────────────────────────────────
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#define ARUCO_NEW_API 1
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#define ARUCO_NEW_API 0
#include <opencv2/aruco.hpp>
#endif

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <iostream>

// ════════════════════════════════════════════════════════════════════════
class ArucoTracker
{
public:
    // A known marker's pose in the world frame
    struct MarkerWorldPose
    {
        int id;
        cv::Mat T_world_marker; // 4×4 homogeneous (double)
    };

    // Result returned per frame
    struct CameraPoseResult
    {
        bool valid = false;
        cv::Mat tvec_world;       // 3×1 camera position in world
        cv::Mat rmat_world;       // 3×3 rotation of camera in world
        cv::Vec3d roll_pitch_yaw; // Euler angles (rad)
        int markers_used = 0;
        double reprojection_error = 0.0;
        std::vector<int> detected_ids;
    };

    /**
     * @param dictionary_name  One of the DICT_* names (e.g. "DICT_4X4_50").
     * @param marker_size      Physical side length of the ArUco marker (m).
     * @param known_markers    Map  id → MarkerWorldPose.
     */
    ArucoTracker(const std::string &dictionary_name,
                 double marker_size,
                 const std::map<int, MarkerWorldPose> &known_markers)
        : marker_size_(marker_size),
          known_markers_(known_markers)
    {
        int dict_id = parseDictionary(dictionary_name);

#if ARUCO_NEW_API
        cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(dict_id);
        cv::aruco::DetectorParameters params;
        detector_ = cv::aruco::ArucoDetector(dict, params);
#else
        dictionary_ = cv::aruco::getPredefinedDictionary(dict_id);
        det_params_ = cv::aruco::DetectorParameters::create();
#endif
    }

    /**
     * Detect ArUco markers, solve per-marker PnP, fuse into a single
     * camera world pose via weighted average.
     *
     * @param frame        BGR image (will be annotated in-place).
     * @param cameraMatrix 3×3 intrinsics.
     * @param distCoeffs   Distortion coefficients.
     * @return             CameraPoseResult (valid == true if ≥1 known marker found).
     */
    CameraPoseResult processFrame(cv::Mat &frame,
                                  const cv::Mat &cameraMatrix,
                                  const cv::Mat &distCoeffs)
    {
        CameraPoseResult result;

        // 1. Detect markers ──────────────────────────────────────────
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;

#if ARUCO_NEW_API
        detector_.detectMarkers(frame, corners, ids, rejected);
#else
        cv::aruco::detectMarkers(frame, dictionary_, corners, ids, det_params_, rejected);
#endif

        if (ids.empty())
            return result;

        result.detected_ids = ids;

        // 1b. Sub-pixel corner refinement ────────────────────────────
        cv::Mat grey;
        if (frame.channels() == 1)
            grey = frame;
        else
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001);
        for (size_t i = 0; i < corners.size(); ++i)
        {
            cv::cornerSubPix(grey, corners[i], cv::Size(11, 11), cv::Size(-1, -1), criteria);
        }

        // Draw markers after refinement so annotations do not affect cornerSubPix.
        cv::aruco::drawDetectedMarkers(frame, corners, ids);

        // 2. Marker-local 3D corners (same for every marker) ────────
        //    Order: top-left, top-right, bottom-right, bottom-left
        //    centred at the marker origin, lying in the XY plane.
        float half = static_cast<float>(marker_size_ / 2.0);
        std::vector<cv::Point3f> obj_pts_local = {
            {-half, half, 0.f},
            {half, half, 0.f},
            {half, -half, 0.f},
            {-half, -half, 0.f}};

        // 3. Per-marker PnP → camera pose in world ──────────────────
        struct PoseEstimate
        {
            cv::Mat T_world_camera; // 4×4
            double weight;          // 1 / reprojection_error
        };
        std::vector<PoseEstimate> estimates;

        for (size_t i = 0; i < ids.size(); ++i)
        {
            int mid = ids[i];
            auto it = known_markers_.find(mid);
            if (it == known_markers_.end())
                continue; // unknown marker

            const cv::Mat &T_world_marker = it->second.T_world_marker;

            // Solve PnP with RANSAC: marker-local frame → camera frame.
            // AP3P is stable for the four ArUco corners in solvePnPRansac.
            const pose_estimation::PnpEstimate pose =
                pose_estimation::solveAp3pRansac(
                    obj_pts_local, corners[i], cameraMatrix, distCoeffs);
            if (!pose.valid || pose.inlier_count < 4)
                continue;

            const double err = pose.mean_reprojection_error;

            // Build T_camera_marker from solvePnP output
            const cv::Mat T_camera_marker = pose_math::makeTransform(
                pose.object_to_camera_rotation,
                pose.object_to_camera_translation);

            // T_world_camera = T_world_marker * inv(T_camera_marker)
            const cv::Mat T_world_camera =
                T_world_marker * pose_math::invertTransform(T_camera_marker);

            double w = (err > 1e-6) ? (1.0 / err) : 1e6;
            estimates.push_back({T_world_camera, w});

            // Draw axis on the frame for this marker
            cv::drawFrameAxes(frame, cameraMatrix, distCoeffs,
                              pose.object_to_camera_rvec,
                              pose.object_to_camera_translation,
                              static_cast<float>(marker_size_ * 0.5));
        }

        if (estimates.empty())
            return result;

        // 4. Weighted average of camera poses ───────────────────────
        //    Average translations directly; average rotations via
        //    rotation vectors and weight them (valid when poses are close).

        double total_weight = 0.0;
        cv::Mat avg_tvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat avg_rvec = cv::Mat::zeros(3, 1, CV_64F);
        double avg_err = 0.0;

        for (const auto &est : estimates)
        {
            cv::Mat R = est.T_world_camera(cv::Rect(0, 0, 3, 3));
            cv::Mat t = est.T_world_camera(cv::Rect(3, 0, 1, 3));
            cv::Mat rv;
            cv::Rodrigues(R, rv);

            avg_tvec += est.weight * t;
            avg_rvec += est.weight * rv;
            avg_err += (1.0 / std::max(est.weight, 1e-12));
            total_weight += est.weight;
        }

        avg_tvec /= total_weight;
        avg_rvec /= total_weight;
        avg_err /= estimates.size();

        cv::Mat avg_rmat;
        cv::Rodrigues(avg_rvec, avg_rmat);

        result.valid = true;
        result.tvec_world = avg_tvec;
        result.rmat_world = avg_rmat;
        result.roll_pitch_yaw = pose_math::rpyFromRotation(avg_rmat);
        result.markers_used = static_cast<int>(estimates.size());
        result.reprojection_error = avg_err;

        return result;
    }

private:
    double marker_size_;
    std::map<int, MarkerWorldPose> known_markers_;

#if ARUCO_NEW_API
    cv::aruco::ArucoDetector detector_;
#else
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> det_params_;
#endif

    // Map string name → OpenCV dictionary ID (works as both enum and int)
    static int parseDictionary(const std::string &name)
    {
        static const std::map<std::string, int> lut = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
            {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
            {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
            {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
            {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
            {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
            {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
            {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
            {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
            {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
            {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
            {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
            {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
        };
        auto it = lut.find(name);
        if (it != lut.end())
            return it->second;

        std::cerr << "Unknown ArUco dictionary '" << name
                  << "', falling back to DICT_4X4_50" << std::endl;
        return cv::aruco::DICT_4X4_50;
    }
};
