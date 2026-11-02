// simulate - trajectories of x' = A x (+ B u), as CORA's linearSys.simulate
//
// Syntax:   x = sys.simulate(problem, alg);
//           x = sys.simulate({.initial_state = x0, .t_final = 2}, alg::Exact{.time_step = 0.1});
// Inputs:   problem - initial_state (start points, columns (..., n, N)), input (a constant input
//                     per trajectory, columns (m, N), or one (m, 1) for all; needs B), t_final
//           alg - alg::Exact (time_step)
// Outputs:  x - x[k] holds the N points at time k*time_step: e^{A k time_step} x0 (exact); with u
//               the states of x' = A x + B u, from e^{M k time_step} [x0; u] with M = [A B; 0 0]
// See also: simulate_random, reach
//
// CORA: contDynamics/@linearSys/simulate.m

#include "cont_dynamics/linear_sys/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// e^{A k dt} x0 for k = 0..steps; the identity map broadcasts x0 to a batch of systems.
std::vector<Tensor> aux_autonomous(const Tensor &A, const Tensor &x0, double time_step, int steps) {
    const Tensor e_adt = (A * time_step).expm();
    Tensor x = A.eye_like().matmul(x0);
    std::vector<Tensor> trajectory{x};
    for (int k = 0; k < steps; ++k) {
        x = e_adt.matmul(x);
        trajectory.push_back(x);
    }
    return trajectory;
}

/// With constant inputs: the input is a state that stays constant, so its effect is exact in
/// one matrix exponential of M = [A B; 0 0].
std::vector<Tensor> aux_with_input(const Tensor &A, const Tensor &B, const Tensor &x0,
                                   const Tensor &u, double time_step, int steps) {
    const int64_t n = A.shape().back(), m = B.shape().back(), N = x0.shape().back();
    const Tensor top = Tensor::cat_last({A, B});
    const Tensor zeros = Tensor::like(A, std::vector<double>(m * (n + m), 0.0), {m, n + m});
    const Tensor E = (Tensor::cat_rows({top, zeros}) * time_step).expm();
    const Tensor ones = Tensor::like(A, std::vector<double>(N, 1.0), {1, N});
    const Tensor u_n = u.shape().back() == N ? u : u.matmul(ones);
    std::vector<int64_t> state_rows(n);
    for (int64_t i = 0; i < n; ++i) state_rows[i] = i;

    Tensor z = Tensor::cat_rows({x0, u_n});
    std::vector<Tensor> trajectory;
    for (int k = 0; k <= steps; ++k) {
        trajectory.push_back(z.transpose().select_cols(state_rows).transpose());
        z = E.matmul(z);
    }
    return trajectory;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> LinearSys::simulate(const SimulationProblem &problem,
                                        const alg::LinearSimulate &alg) const {
    detail::Entry entry;
    const Tensor &x0 = problem.initial_state;
    CORA_CHECK(x0.defined() && x0.shape().size() >= 2 && x0.shape()[x0.shape().size() - 2] == dim(),
               Error::dimension_mismatch, [&] {
                   return "simulate: the initial states must be columns (..., " +
                          std::to_string(dim()) + ", N), got the shape " +
                          (x0.defined() ? shape_text(x0.shape()) : std::string("(none)"));
               });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "simulate: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    if (entry.check()) std::visit([](const auto &a) { a.validate(); }, alg);
    const double time_step = std::get<alg::Exact>(alg).time_step;
    const int steps = priv_num_steps(problem.t_final, time_step);
    if (!problem.input) return aux_autonomous(A_, x0, time_step, steps);

    const Tensor &u = *problem.input;
    CORA_CHECK(B_.has_value(), Error::wrong_value,
               [] { return std::string("simulate: an input needs B; use LinearSys(A, B)"); });
    CORA_CHECK(u.shape().size() == 2 && u.shape()[0] == B_->shape().back() &&
                   (u.shape().back() == x0.shape().back() || u.shape().back() == 1),
               Error::dimension_mismatch, [&] {
                   return "simulate: the input must be (m, N) or (m, 1) with m = " +
                          std::to_string(B_->shape().back()) + " and N = " +
                          std::to_string(x0.shape().back()) + ", got " + shape_text(u.shape());
               });
    return aux_with_input(A_, *B_, x0, u, time_step, steps);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
