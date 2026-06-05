#pragma once
/**
 * Position Kalman Filter for PnP-based marker localisation.
 *
 * Design rationale — KF vs EKF:
 *   The PnP solver outputs 3-D translation (x, y, z) directly in camera
 *   frame.  Both the state-transition model (constant-velocity kinematics)
 *   and the observation model (direct position readout) are *linear*, so a
 *   standard Kalman Filter is optimal in the minimum-variance sense.
 *
 * State vector:  x = [px, py, pz, vx, vy, vz]^T   (6 × 1)
 * Measurement:   z = [px, py, pz]^T                 (3 × 1)
 */

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class PositionKalmanFilter {
public:
    /**
     * @param process_noise_std   Std-dev of acceleration noise (m/s²).
     *                            Larger → trusts measurements more.
     * @param measurement_noise_std  Std-dev of PnP measurement noise (m).
     */
    PositionKalmanFilter(double process_noise_std = 0.5,
                         double measurement_noise_std = 0.02)
        : process_noise_std_(process_noise_std),
          measurement_noise_std_(measurement_noise_std),
          initialized_(false),
          last_time_(0.0)
    {
        x_.setZero();                     // state
        P_ = Eigen::Matrix<double,6,6>::Identity(); // covariance

        // Measurement matrix  H:  z = H * x
        H_.setZero();
        H_(0, 0) = 1.0;
        H_(1, 1) = 1.0;
        H_(2, 2) = 1.0;

        // Measurement noise covariance
        R_ = Eigen::Matrix3d::Identity() * (measurement_noise_std * measurement_noise_std);
    }

    /** Reset to uninitialised state. */
    void reset() {
        initialized_ = false;
        x_.setZero();
        P_ = Eigen::Matrix<double,6,6>::Identity();
        last_time_ = 0.0;
    }

    /**
     * Feed a new PnP position measurement.
     *
     * @param measurement  Measured [x, y, z] in metres.
     * @param timestamp    Monotonic timestamp in seconds.
     * @return             Filtered position [x, y, z].
     */
    Eigen::Vector3d update(const Eigen::Vector3d& measurement, double timestamp) {
        if (!initialized_) {
            x_.head<3>() = measurement;
            last_time_   = timestamp;
            initialized_ = true;
            return measurement;
        }

        double dt = timestamp - last_time_;
        if (dt <= 0.0) {
            // Duplicate / out-of-order — return current estimate
            return x_.head<3>();
        }
        last_time_ = timestamp;

        // ── Predict ──────────────────────────────────────────────
        auto F = transitionMatrix(dt);
        auto Q = processNoiseMatrix(dt);

        Eigen::Matrix<double,6,1> x_pred = F * x_;
        Eigen::Matrix<double,6,6> P_pred = F * P_ * F.transpose() + Q;

        // ── Update ───────────────────────────────────────────────
        Eigen::Vector3d y = measurement - H_ * x_pred;             // innovation
        Eigen::Matrix3d S = H_ * P_pred * H_.transpose() + R_;     // innovation cov
        Eigen::Matrix<double,6,3> K = P_pred * H_.transpose() * S.inverse(); // gain

        x_ = x_pred + K * y;
        P_ = (Eigen::Matrix<double,6,6>::Identity() - K * H_) * P_pred;

        return x_.head<3>();
    }

    /**
     * Predict-only step (no measurement, e.g. marker lost).
     * Useful for maintaining an estimate during dropouts.
     */
    Eigen::Vector3d predictOnly(double timestamp) {
        if (!initialized_) return Eigen::Vector3d::Zero();

        double dt = timestamp - last_time_;
        if (dt <= 0.0) return x_.head<3>();
        last_time_ = timestamp;

        auto F = transitionMatrix(dt);
        auto Q = processNoiseMatrix(dt);

        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q;

        return x_.head<3>();
    }

    Eigen::Vector3d position()  const { return x_.head<3>(); }
    Eigen::Vector3d velocity()  const { return x_.tail<3>(); }
    bool isInitialized()        const { return initialized_; }

private:
    /** Constant-velocity state transition matrix. */
    Eigen::Matrix<double,6,6> transitionMatrix(double dt) const {
        Eigen::Matrix<double,6,6> F = Eigen::Matrix<double,6,6>::Identity();
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;
        return F;
    }

    /**
     * Discrete process-noise covariance assuming piece-wise constant
     * acceleration with std-dev  process_noise_std_.
     */
    Eigen::Matrix<double,6,6> processNoiseMatrix(double dt) const {
        double q   = process_noise_std_ * process_noise_std_;
        double dt2 = dt  * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;

        Eigen::Matrix<double,6,6> Q = Eigen::Matrix<double,6,6>::Zero();
        for (int i = 0; i < 3; ++i) {
            Q(i,     i)     = q * dt4 / 4.0;
            Q(i,     i + 3) = q * dt3 / 2.0;
            Q(i + 3, i)     = q * dt3 / 2.0;
            Q(i + 3, i + 3) = q * dt2;
        }
        return Q;
    }

    double process_noise_std_;
    double measurement_noise_std_;
    bool   initialized_;
    double last_time_;

    Eigen::Matrix<double,6,1> x_;       // state [px py pz vx vy vz]
    Eigen::Matrix<double,6,6> P_;       // covariance
    Eigen::Matrix<double,3,6> H_;       // measurement matrix
    Eigen::Matrix3d           R_;       // measurement noise covariance
};
