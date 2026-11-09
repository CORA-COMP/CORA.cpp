// simulate - trajectories of x' = f(x, u), as CORA's nonlinearSys.simulate
//
// Syntax:   x = sys.simulate(problem, alg);
//           x = sys.simulate({.initial_state = x0, .t_final = 1}, alg::RungeKutta4{});
// Inputs:   problem - initial_state (columns (n, N)), input (constant: (m, N) or (m, 1)), t_final
//           alg - alg::RungeKutta4: time_step (the sampling), max_step (the largest step, 1e-3)
// Outputs:  x - x[k] holds the N points at time k*time_step, from the classical Runge-Kutta method
// See also: simulate_random, reach
//
// CORA: contDynamics/@nonlinearSys/simulate.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/tensor/tolerance.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// One step of size h of the classical Runge-Kutta method, for all columns of x at once.
Tensor aux_runge_kutta(const NonlinearSys &sys, const Tensor &x, const std::optional<Tensor> &u,
                       double h) {
    const Tensor k1 = sys.dynamics(x, u);
    const Tensor k2 = sys.dynamics(x + k1 * (h / 2), u);
    const Tensor k3 = sys.dynamics(x + k2 * (h / 2), u);
    const Tensor k4 = sys.dynamics(x + k3 * h, u);
    return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> NonlinearSys::simulate(const SimulationProblem &problem,
                                           const alg::NonlinearSimulate &alg) const {
    detail::Entry entry;
    // columns of start points, an input of the inputs, a positive horizon
    const Tensor &x0 = problem.initial_state;
    CORA_CHECK(x0.defined() && x0.shape().size() == 2 && x0.shape()[0] == n_, Error::not_supported,
               [&] {
                   return "simulate: the start points of a nonlinear system are columns (" +
                          std::to_string(n_) + ", N); batches are not supported";
               });
    const std::optional<Tensor> &u = problem.input;
    CORA_CHECK(m_ > 0 || !u, Error::not_supported,
               [] { return std::string("simulate: the system has no inputs; leave input empty"); });
    CORA_CHECK(m_ == 0 || (u && u->shape().size() == 2 && u->shape()[0] == m_ &&
                           (u->shape()[1] == 1 || u->shape()[1] == x0.shape()[1])),
               Error::dimension_mismatch, [&] {
                   return "simulate: the input must be columns (" + std::to_string(m_) +
                          ", N) or one column (" + std::to_string(m_) + ", 1)";
               });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "simulate: t_final must be positive and finite, got " +
               std::to_string(problem.t_final);
    });
    // the options, then every step
    if (entry.check()) std::visit([](const auto &a) { a.validate(); }, alg);
    // every step of time_step in substeps no longer than max_step
    const alg::RungeKutta4 &options = std::get<alg::RungeKutta4>(alg);
    const double time_step = options.time_step;
    const int steps = priv_num_steps(problem.t_final, time_step);
    const int substeps =
        static_cast<int>(std::ceil(time_step / options.max_step - numerics::kStepCount));
    const double h = time_step / substeps;

    std::vector<Tensor> trajectory{x0};
    Tensor x = x0;
    for (int k = 0; k < steps; ++k) {
        for (int s = 0; s < substeps; ++s) x = aux_runge_kutta(*this, x, u, h);
        trajectory.push_back(x);
    }
    return trajectory;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
