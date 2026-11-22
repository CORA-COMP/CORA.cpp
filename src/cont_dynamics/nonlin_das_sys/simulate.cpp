// simulate - a trajectory of the differential-algebraic system by the classical Runge-Kutta method
//
// The system has index 1, so y follows from x by g(x, y, u) = 0. At every stage of the method y is
// solved by Newton's method from the value of the stage before, and the stage's slope is f there.
// CORA calls ode15s on the mass-matrix form; this is the fixed-step method of NonlinearSys.
//
// Syntax:   DaeTrajectory T = sys.simulate(x0, y_guess, u, t_final, alg);
// Inputs:   x0 - the start state (n), y_guess - a guess of y at it (p), u - the constant input
//           (m), t_final - the horizon, alg - alg::RungeKutta4: the sampling and integration steps
// Outputs:  T - x (K+1, n) and y (K+1, p) at the sample times k * time_step
// See also: NonlinearSys::simulate, consistent_initial_state
//
// CORA: contDynamics/@nonlinDASys/simulate.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   21-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

DaeTrajectory NonlinDASys::simulate(const Tensor &x0, const Tensor &y_guess, const Tensor &u,
                                    double t_final, const alg::NonlinearSimulate &alg) const {
    detail::Entry entry;
    CORA_CHECK(m_ == 0 || (u.defined() && u.shape() == std::vector<int64_t>({m_})),
               Error::dimension_mismatch,
               [&] { return "simulate: the input must be a vector (" + std::to_string(m_) + ")"; });
    // the recursion runs on the columns of CORA
    const Tensor x0_col = x0.defined() ? x0.expand_dims(-1) : x0;
    const Tensor y_col = y_guess.defined() ? y_guess.expand_dims(-1) : y_guess;
    const Tensor u_col = u.defined() ? u.expand_dims(-1) : u;
    if (entry.check()) validate_state("simulate", x0, y_guess);
    CORA_CHECK(t_final > 0 && t_final < 1e300, Error::wrong_value, [&] {
        return "simulate: t_final must be positive and finite, got " + std::to_string(t_final);
    });
    if (entry.check()) std::visit([](const auto &a) { a.validate(); }, alg);
    // every sample step in substeps no longer than max_step
    const alg::RungeKutta4 &options = std::get<alg::RungeKutta4>(alg);
    const int steps = priv_num_steps(t_final, options.time_step);
    const int substeps =
        static_cast<int>(std::ceil(options.time_step / options.max_step - numerics::kStepCount));
    const double h = options.time_step / substeps;
    // the classical method, y solved at the state of every stage from its last value
    const auto aux_slope = [&](const Tensor &x_, Tensor &y_) {
        y_ = newton_algebraic(x_, y_, u_col, true);
        return dynamics_columns(x_, y_, u_col);
    };
    Tensor x = x0_col, y = newton_algebraic(x0_col, y_col, u_col, true);
    std::vector<Tensor> xs{x}, ys{y};
    for (int k = 0; k < steps; ++k) {
        for (int s = 0; s < substeps; ++s) {
            const Tensor k1 = aux_slope(x, y);
            const Tensor k2 = aux_slope(x + k1 * (h / 2), y);
            const Tensor k3 = aux_slope(x + k2 * (h / 2), y);
            const Tensor k4 = aux_slope(x + k3 * h, y);
            x = x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6);
        }
        y = newton_algebraic(x, y, u_col, true);
        xs.push_back(x);
        ys.push_back(y);
    }
    return {Tensor::concat(xs, -1).matrix_transpose(), Tensor::concat(ys, -1).matrix_transpose()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
