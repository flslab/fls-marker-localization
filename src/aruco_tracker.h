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

// ─── ArUco API version detection ────────────────────────────────────────
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#define ARUCO_NEW_API 1
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#define ARUCO_NEW_API 0
#include <opencv2/aruco.hpp>
#endif

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

// ─── helper: euler (ZYX intrinsic) → 3×3 rotation matrix ───────────────
static cv::Mat eulerToRotationMatrix(double roll_rad, double pitch_rad, double yaw_rad)
{
    double cr = std::cos(roll_rad), sr = std::sin(roll_rad);
    double cp = std::cos(pitch_rad), sp = std::sin(pitch_rad);
    double cy = std::cos(yaw_rad), sy = std::sin(yaw_rad);

    cv::Mat R = (cv::Mat_<double>(3, 3) << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                 sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                 -sp, cp * sr, cp * cr);
    return R;
}

// ─── helper: build a 4×4 homogeneous transform ─────────────────────────
static cv::Mat makeTransform(const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    t.reshape(1, 3).copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

// ─── helper: invert a rigid 4×4 transform ──────────────────────────────
static cv::Mat invertTransform(const cv::Mat &T)
{
    cv::Mat R = T(cv::Rect(0, 0, 3, 3));
    cv::Mat t = T(cv::Rect(3, 0, 1, 3));
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * t;

    cv::Mat T_inv = cv::Mat::eye(4, 4, CV_64F);
    R_inv.copyTo(T_inv(cv::Rect(0, 0, 3, 3)));
    t_inv.copyTo(T_inv(cv::Rect(3, 0, 1, 3)));
    return T_inv;
}

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
        cv::Vec3d yaw_pitch_roll; // Euler angles (rad)
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
        struct PnPCandidate
        {
            cv::Mat rvec;
            cv::Mat tvec;
            cv::Mat T_world_camera;
            double reprojection_error = std::numeric_limits<double>::infinity();
            double continuity_cost = std::numeric_limits<double>::infinity();
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
            std::vector<PnPCandidate> candidates;
            auto addCandidate = [&](const cv::Mat &candidate_rvec, const cv::Mat &candidate_tvec)
            {
                if (candidate_rvec.empty() || candidate_tvec.empty())
                    return;

                cv::Mat R_cm;
                cv::Rodrigues(candidate_rvec, R_cm);
                cv::Mat T_camera_marker = makeTransform(R_cm, candidate_tvec);
                cv::Mat T_world_camera = T_world_marker * invertTransform(T_camera_marker);

                double err = computeReprojectionError(obj_pts_local, corners[i],
                                                      candidate_rvec, candidate_tvec,
                                                      cameraMatrix, distCoeffs);
                double continuity = has_last_pose_
                    ? poseContinuityCost(T_world_camera, last_T_world_camera_, marker_size_)
                    : 0.0;

                candidates.push_back({candidate_rvec.clone(), candidate_tvec.clone(),
                                      T_world_camera.clone(), err, continuity});
            };

            cv::Mat ransac_rvec, ransac_tvec, inliers;
            bool ransac_ok = cv::solvePnPRansac(obj_pts_local, corners[i],
                                                cameraMatrix, distCoeffs,
                                                ransac_rvec, ransac_tvec, false,
                                                100, 4.0f, 0.99, inliers,
                                                cv::SOLVEPNP_AP3P);
            if (ransac_ok && inliers.total() >= 4)
                addCandidate(ransac_rvec, ransac_tvec);

            // Planar square markers can have two very similar reprojection solutions.
            // Expose both and pick the one that is temporally consistent.
            std::vector<cv::Mat> generic_rvecs, generic_tvecs;
            try
            {
                cv::solvePnPGeneric(obj_pts_local, corners[i],
                                    cameraMatrix, distCoeffs,
                                    generic_rvecs, generic_tvecs,
                                    false, cv::SOLVEPNP_IPPE_SQUARE);
                for (size_t c = 0; c < generic_rvecs.size(); ++c)
                    addCandidate(generic_rvecs[c], generic_tvecs[c]);
            }
            catch (const cv::Exception &)
            {
                // Keep the RANSAC solution if IPPE cannot provide candidates.
            }

            if (candidates.empty())
                continue;

            const auto best_err_it = std::min_element(
                candidates.begin(), candidates.end(),
                [](const PnPCandidate &a, const PnPCandidate &b)
                {
                    return a.reprojection_error < b.reprojection_error;
                });

            auto selected_it = best_err_it;
            if (has_last_pose_ && candidates.size() > 1)
            {
                const double max_extra_reprojection_error_px = 2.0;
                const double reprojection_limit =
                    best_err_it->reprojection_error + max_extra_reprojection_error_px;

                selected_it = std::min_element(
                    candidates.begin(), candidates.end(),
                    [reprojection_limit](const PnPCandidate &a, const PnPCandidate &b)
                    {
                        const bool a_allowed = a.reprojection_error <= reprojection_limit;
                        const bool b_allowed = b.reprojection_error <= reprojection_limit;
                        if (a_allowed != b_allowed)
                            return a_allowed;
                        if (a_allowed)
                            return a.continuity_cost < b.continuity_cost;
                        return a.reprojection_error < b.reprojection_error;
                    });
            }

            const PnPCandidate &selected = *selected_it;

            double w = (selected.reprojection_error > 1e-6)
                ? (1.0 / selected.reprojection_error)
                : 1e6;
            estimates.push_back({selected.T_world_camera, w});

            // Draw axis on the frame for this marker
            cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, selected.rvec, selected.tvec,
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
        result.yaw_pitch_roll = yawPitchRollDecomposition(avg_rmat);
        result.markers_used = static_cast<int>(estimates.size());
        result.reprojection_error = avg_err;

        last_T_world_camera_ = makeTransform(avg_rmat, avg_tvec).clone();
        has_last_pose_ = true;

        return result;
    }

private:
    double marker_size_;
    std::map<int, MarkerWorldPose> known_markers_;
    bool has_last_pose_ = false;
    cv::Mat last_T_world_camera_;

#if ARUCO_NEW_API
    cv::aruco::ArucoDetector detector_;
#else
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> det_params_;
#endif

    // Euler decomposition (same convention as the rest of the codebase)
    static cv::Vec3d yawPitchRollDecomposition(const cv::Mat &rmat)
    {
        double yaw = std::atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
        double pitch = std::atan2(-rmat.at<double>(2, 0),
                                  std::sqrt(std::pow(rmat.at<double>(2, 1), 2) +
                                            std::pow(rmat.at<double>(2, 2), 2)));
        double roll = std::atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));
        return cv::Vec3d(yaw, pitch, roll);
    }

    static double computeReprojectionError(const std::vector<cv::Point3f> &object_points,
                                           const std::vector<cv::Point2f> &image_points,
                                           const cv::Mat &rvec,
                                           const cv::Mat &tvec,
                                           const cv::Mat &cameraMatrix,
                                           const cv::Mat &distCoeffs)
    {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points, rvec, tvec, cameraMatrix, distCoeffs, projected);

        double err = 0.0;
        for (size_t i = 0; i < image_points.size(); ++i)
            err += cv::norm(image_points[i] - projected[i]);

        return image_points.empty() ? std::numeric_limits<double>::infinity()
                                    : err / static_cast<double>(image_points.size());
    }

    static double rotationDistanceRad(const cv::Mat &Ra, const cv::Mat &Rb)
    {
        cv::Mat R_delta = Ra * Rb.t();
        double trace = R_delta.at<double>(0, 0) +
                       R_delta.at<double>(1, 1) +
                       R_delta.at<double>(2, 2);
        double cos_angle = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
        return std::acos(cos_angle);
    }

    static double poseContinuityCost(const cv::Mat &T_world_camera,
                                     const cv::Mat &last_T_world_camera,
                                     double translation_scale)
    {
        cv::Mat R = T_world_camera(cv::Rect(0, 0, 3, 3));
        cv::Mat last_R = last_T_world_camera(cv::Rect(0, 0, 3, 3));
        cv::Mat t = T_world_camera(cv::Rect(3, 0, 1, 3));
        cv::Mat last_t = last_T_world_camera(cv::Rect(3, 0, 1, 3));

        double scale = std::max(std::abs(translation_scale), 1e-6);
        double translation_cost = cv::norm(t - last_t) / scale;
        double rotation_cost = rotationDistanceRad(R, last_R);

        return translation_cost + rotation_cost;
    }

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
