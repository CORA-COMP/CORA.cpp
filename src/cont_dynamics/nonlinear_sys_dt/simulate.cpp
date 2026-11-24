// simulate - trajectories of x(k+1) = f(x(k), u(k)), as CORA's nonlinearSysDT.simulate
//
// The recursion is evaluated on all start points at once. The input is constant (CORA's params.u a
// column) or one input per step (a matrix with a column per step; one more if the outputs are
// asked, as the output y(k) reads u(k)). Without an input, the input 0.
//
// Syntax:   T = sys.simulate({.initial_state = x0, .t_final = 1});
// Inputs:   problem - initial_state (N, n), input or input_signal, t_final (a multiple of dt)
// Outputs:  T - x[k] (N, n) at time k*dt, and y[k] (N, r) if asked
// See also: reach
//
// CORA: contDynamics/@nonlinearSysDT/simulate.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys_dt/nonlinear_sys_dt.h"
#include "cora/global/tensor/tolerance.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

DiscreteTrajectory NonlinearSysDT::simulate(const DiscreteSimulationProblem &problem) const {
    detail::Entry entry;
    const Tensor &x0 = problem.initial_state;
    const int64_t m = input_dim();
    CORA_CHECK(x0.defined() && x0.shape().size() == 2 && x0.shape()[1] == dim(),
               Error::dimension_mismatch, [&] {
                   return "simulate: the start points must be rows (N, " + std::to_string(dim()) +
                          ")";
               });
    CORA_CHECK(m > 0 || (!problem.input && !problem.input_signal), Error::not_supported,
               [] { return std::string("simulate: the system has no inputs; leave input empty"); });
    CORA_CHECK(!(problem.input && problem.input_signal), Error::wrong_value,
               [] { return std::string("simulate: give input or input_signal, not both"); });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "simulate: t_final must be positive and finite, got " +
               std::to_string(problem.t_final);
    });
    // the final time is a multiple of the sampling time
    const int steps =
        static_cast<int>(std::floor(problem.t_final / time_step_ + numerics::kStepCount));
    CORA_CHECK(std::abs(steps * time_step_ - problem.t_final) <= numerics::kStepCount * time_step_,
               Error::wrong_value, [] {
                   return std::string("simulate: the final time has to be a multiple of the "
                                      "sampling time");
               });
    const std::size_t needed = std::size_t(steps) + (problem.with_output ? 1 : 0);
    CORA_CHECK(!problem.input_signal || problem.input_signal->size() == needed,
               Error::dimension_mismatch, [&] {
                   return "simulate: input_signal must hold one input per " +
                          std::string(problem.with_output ? "time point" : "step") + " (" +
                          std::to_string(needed) + ")";
               });
    // the input at step k: the signal's, the constant one, or 0
    const auto input_at = [&](int k) -> std::optional<Tensor> {
        if (m == 0) return std::nullopt;
        if (problem.input_signal) return (*problem.input_signal)[std::size_t(k)];
        if (problem.input) return *problem.input;
        return x0.zeros_like({m});
    };
    DiscreteTrajectory T;
    Tensor x = x0;
    T.x.push_back(x);
    if (problem.with_output) T.y.push_back(output(x, input_at(0)));
    for (int k = 0; k < steps; ++k) {
        x = next_state(x, input_at(k));
        T.x.push_back(x);
        if (problem.with_output) T.y.push_back(output(x, input_at(k + 1)));
    }
    return T;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
