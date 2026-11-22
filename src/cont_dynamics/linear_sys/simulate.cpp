// simulate - trajectories of x' = A x (+ B u), as CORA's linearSys.simulate
//
// input is a constant input per trajectory, rows (..., N, m), or one vector (m) for all. With u the
// states of x' = A x + B u come from e^{M k time_step} [x0; u] with M = [A B; 0 0].
//
// Syntax:   x = sys.simulate(problem, alg);
//           x = sys.simulate({.initial_state = x0, .t_final = 2}, alg::Exact{.time_step = 0.1});
// Inputs:   problem - initial_state (..., N, n), input (needs B), t_final
//           alg - alg::Exact (time_step)
// Outputs:  x - x[k] holds the N points (..., N, n) at time k*time_step: e^{A k time_step} x0
// See also: simulate_random, reach
//
// CORA: contDynamics/@linearSys/simulate.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   21-November-2026
// Reviewed:      ---

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
    const Tensor top = Tensor::concat({A, B}, -1);
    const Tensor zeros = A.zeros_like({m, n + m});
    const Tensor E = (Tensor::concat({top, zeros}, -2) * time_step).expm();
    const Tensor ones = A.ones_like({1, N});
    const Tensor u_n = u.shape().back() == N ? u : u.matmul(ones);
    std::vector<int64_t> state_rows(n);
    for (int64_t i = 0; i < n; ++i) state_rows[i] = i;

    Tensor z = Tensor::concat({x0, u_n}, -2);
    std::vector<Tensor> trajectory;
    for (int k = 0; k <= steps; ++k) {
        trajectory.push_back(z.matrix_transpose().take(state_rows, -1).matrix_transpose());
        z = E.matmul(z);
    }
    return trajectory;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> LinearSys::simulate(const SimulationProblem &problem,
                                        const alg::LinearSimulate &alg) const {
    detail::Entry entry;
    // the start points, the horizon and the algorithm
    const Tensor &points = problem.initial_state;
    CORA_CHECK(points.defined() && points.shape().size() >= 2 && points.shape().back() == dim(),
               Error::dimension_mismatch, [&] {
                   return "simulate: the initial states must be rows (..., N, " +
                          std::to_string(dim()) + "), got the shape " +
                          (points.defined() ? shape_text(points.shape()) : std::string("(none)"));
               });
    // the recursion runs on the columns (..., n, N) of CORA, and gives rows back
    const Tensor x0 = points.matrix_transpose();
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "simulate: t_final must be positive and finite, got " +
               std::to_string(problem.t_final);
    });
    if (entry.check()) std::visit([](const auto &a) { a.validate(); }, alg);
    const double time_step = std::get<alg::Exact>(alg).time_step;
    const int steps = priv_num_steps(problem.t_final, time_step);
    const auto rows = [](std::vector<Tensor> columns) {
        for (Tensor &x : columns) x = x.matrix_transpose();
        return columns;
    };
    if (!problem.input) return rows(aux_autonomous(A(), x0, time_step, steps));

    // a constant input: B, and one row per trajectory or one vector for all
    const Tensor &u = *problem.input;
    CORA_CHECK(B_.has_value(), Error::wrong_value,
               [] { return std::string("simulate: an input needs B; use LinearSys(A, B)"); });
    const int64_t m = B_->shape().back();
    CORA_CHECK(u.defined() && ((u.shape().size() == 1 && u.shape()[0] == m) ||
                               (u.shape().size() == 2 && u.shape()[1] == m &&
                                (u.shape()[0] == x0.shape().back() || u.shape()[0] == 1))),
               Error::dimension_mismatch, [&] {
                   return "simulate: the input must be (N, m) or (m) with m = " +
                          std::to_string(m) + " and N = " + std::to_string(x0.shape().back()) +
                          ", got " + (u.defined() ? shape_text(u.shape()) : std::string("(none)"));
               });
    // one column per trajectory, or one for all
    const Tensor u_columns = u.shape().size() == 1 ? u.expand_dims(-1) : u.matrix_transpose();
    return rows(aux_with_input(A(), *B_, x0, u_columns, time_step, steps));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
