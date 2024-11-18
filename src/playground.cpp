#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/rotation.h>

// Cost function for Ceres optimization
struct EllipseToSphereCost {
    EllipseToSphereCost(const ImplicitEllipse& ellipse, double r)
        : ellipse_(ellipse), r_(r) {}

    template <typename T>
    bool operator()(const T* const X_Var, T* residuals) const {
        // X_Var: [S0_x, S0_y, S0_z, alpha]
        T S0_x = X_Var[0];
        T S0_y = X_Var[1];
        T S0_z = X_Var[2];
        T alpha = X_Var[3];

        // Implicit ellipse parameters
        T A = T(ellipse_.A);
        T B = T(ellipse_.B);
        T C = T(ellipse_.C);
        T D = T(ellipse_.D);
        T E = T(ellipse_.E);
        T F = T(ellipse_.F);

        // Constant matrix as per MATLAB code
        T constMatrix[6][7] = {
            {T(0), T(1), T(1), T(0), T(0), T(0), T(-r_ * r_)},
            {T(0), T(0), T(0), T(-2), T(0), T(0), T(0)},
            {T(1), T(0), T(1), T(0), T(0), T(0), T(-r_ * r_)},
            {T(0), T(0), T(0), T(0), T(-2), T(0), T(0)},
            {T(0), T(0), T(0), T(0), T(0), T(-2), T(0)},
            {T(1), T(1), T(0), T(0), T(0), T(0), T(-r_ * r_)}
        };

        // Compute the residuals (difference between implicit ellipse and sphere equations)
        residuals[0] = A - (constMatrix[0][0] * alpha * S0_x * S0_x +
                            constMatrix[0][1] * alpha * S0_y * S0_y +
                            constMatrix[0][2] * alpha * S0_z * S0_z +
                            constMatrix[0][6] * alpha);
        residuals[1] = B - (constMatrix[1][3] * alpha * S0_x * S0_y);
        residuals[2] = C - (constMatrix[2][0] * alpha * S0_x * S0_x +
                            constMatrix[2][2] * alpha * S0_z * S0_z +
                            constMatrix[2][6] * alpha);
        residuals[3] = D - (constMatrix[3][3] * alpha * S0_x);
        residuals[4] = E - (constMatrix[4][4] * alpha * S0_y);
        residuals[5] = F - (constMatrix[5][0] * alpha * S0_x * S0_x +
                            constMatrix[5][1] * alpha * S0_y * S0_y +
                            constMatrix[5][6] * alpha);
        return true;
    }

private:
    ImplicitEllipse ellipse_;
    double r_;
};

// Function to perform the Levenberg-Marquardt optimization
std::pair<std::array<double, 3>, double> implEllipse2implSphereOpt(
    const ImplicitEllipse& ellipse, double r, std::array<double, 3> S0, double alpha) {

    // Initial values for S0 (sphere center) and alpha (scale)
    double X_Var[4] = {S0[0], S0[1], S0[2], alpha};

    // Set up the problem
    ceres::Problem problem;

    // Add the residual block (cost function)
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<EllipseToSphereCost, 6, 4>(
            new EllipseToSphereCost(ellipse, r)),
        nullptr,  // No loss function
        X_Var);

    // Set bounds for the optimization (equivalent to MATLAB's LB and UB)
    problem.SetParameterLowerBound(X_Var, 0, -5);   // S0_x lower bound
    problem.SetParameterUpperBound(X_Var, 0, 5);    // S0_x upper bound
    problem.SetParameterLowerBound(X_Var, 1, -5);   // S0_y lower bound
    problem.SetParameterUpperBound(X_Var, 1, 5);    // S0_y upper bound
    problem.SetParameterLowerBound(X_Var, 2, 0);    // S0_z lower bound
    problem.SetParameterUpperBound(X_Var, 2, 25);   // S0_z upper bound
    problem.SetParameterLowerBound(X_Var, 3, 0);    // alpha lower bound
    problem.SetParameterUpperBound(X_Var, 3, 10000);  // alpha upper bound

    // Set solver options (equivalent to MATLAB's 'optimoptions')
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 3000;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1e-20;
    options.parameter_tolerance = 1e-12;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Output the optimization result
    std::cout << summary.FullReport() << "\n";

    // Return the optimized values for S0 and alpha
    return {{X_Var[0], X_Var[1], X_Var[2]}, X_Var[3]};
}
