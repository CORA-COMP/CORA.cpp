// simulate - trajectories of x' = f(x, u), as CORA's nonlinearSys.simulate
//
// Syntax:   x = sys.simulate(problem, alg);
//           x = sys.simulate({.initial_state = x0, .t_final = 1}, alg::RungeKutta4{});
// Inputs:   problem - initial_state (N, n), input (constant: (N, m) or (m)), t_final
//           alg - alg::RungeKutta4: time_step (the sampling), max_step (the largest step, 1e-3)
// Outputs:  x - x[k] holds the N points (N, n) at time k*time_step (classical Runge-Kutta)
// See also: simulate_random, reach
//
// CORA: contDynamics/@nonlinearSys/simulate.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   21-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/tensor/tolerance.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// One step of size h of the classical Runge-Kutta method for the slope f, for all columns of x.
template <class Slope>
Tensor aux_runge_kutta(const Slope &f, const Tensor &x, double h) {
    const Tensor k1 = f(x);
    const Tensor k2 = f(x + k1 * (h / 2));
    const Tensor k3 = f(x + k2 * (h / 2));
    const Tensor k4 = f(x + k3 * h);
    return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> NonlinearSys::simulate(const SimulationProblem &problem,
                                           const alg::NonlinearSimulate &alg) const {
    detail::Entry entry;
    // rows of start points, an input of the inputs, a positive horizon
    const Tensor &points = problem.initial_state;
    CORA_CHECK(points.defined() && points.shape().size() == 2 && points.shape()[1] == n_,
               Error::not_supported, [&] {
                   return "simulate: the start points of a nonlinear system are rows (N, " +
                          std::to_string(n_) + "); batches are not supported";
               });
    const std::optional<Tensor> &input = problem.input;
    CORA_CHECK(m_ > 0 || !input, Error::not_supported,
               [] { return std::string("simulate: the system has no inputs; leave input empty"); });
    CORA_CHECK(m_ == 0 || (input && input->defined() &&
                           ((input->shape().size() == 1 && input->shape()[0] == m_) ||
                            (input->shape().size() == 2 && input->shape()[1] == m_ &&
                             (input->shape()[0] == 1 || input->shape()[0] == points.shape()[0])))),
               Error::dimension_mismatch, [&] {
                   return "simulate: the input must be rows (N, " + std::to_string(m_) +
                          ") or one vector (" + std::to_string(m_) + ")";
               });
    // the recursion runs on the columns of CORA, and gives rows back
    const Tensor x0 = points.matrix_transpose();
    const std::optional<Tensor> u =
        !input ? std::nullopt
               : std::optional<Tensor>(input->shape().size() == 1 ? input->expand_dims(-1)
                                                                  : input->matrix_transpose());
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

    const auto slope = [&](const Tensor &x_) { return dynamics_columns(x_, u); };
    std::vector<Tensor> trajectory{points};
    Tensor x = x0;
    for (int k = 0; k < steps; ++k) {
        for (int s = 0; s < substeps; ++s) x = aux_runge_kutta(slope, x, h);
        trajectory.push_back(x.matrix_transpose());
    }
    return trajectory;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
