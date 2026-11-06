// test_linear_sys_steps_property - the steps of reach enclose the trajectories
//
// On random systems and sets with fixed seeds, the exact state of every sampled start point and
// constant input lies in the sets of every solution of a step, at its end and inside it.
//
// PROPERTY: contDynamics/@linearSys/homogeneousSolution.m
// PROPERTY: contDynamics/@linearSys/particularSolution_timeVarying.m
// PROPERTY: contDynamics/@linearSys/particularSolution_constant.m
// PROPERTY: contDynamics/@linearSys/affineSolution.m
// PROPERTY: contDynamics/@linearSys/oneStep.m
// PROPERTY: contDynamics/@linearSys/initReach.m
// PROPERTY: contDynamics/@linearSys/post.m
// PROPERTY: contDynamics/@linearSys/errorSolution_adaptive.m
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"
#include "cora/global/rng.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kSystems = 6, kPoints = 8;

/// The state at t of x' = A x + w from x0, by the matrix exponential of [A w; 0 0].
Tensor exact_state(const Tensor &A, const Tensor &x0, const Tensor &w, double t) {
    const int64_t n = A.shape()[0];
    const Tensor top = Tensor::concat({A, w}, -1);
    const Tensor M = Tensor::concat({top, A.zeros_like({1, n + 1})}, -2);
    const Tensor start = Tensor::concat({x0, A.ones_like({1, 1})}, -2);
    return (M * t).expm().matmul(start).slice(-2, 0, n);
}

/// A random stable system with inputs, sets, sample points and one constant input of U.
struct Case {
    LinearSys sys;
    Zonotope X, U;
    Tensor u, points, inputs;
};

Case random_case(cora::Rng &rng) {
    const LinearSys sys = LinearSys::generate_random(rng, 3, 3, 1);
    const Tensor eye = Tensor::eye(3);
    const Zonotope X = Zonotope::generate_random(rng, 3, std::nullopt, 4);
    // a centered input set in the state space, and a constant input u
    const Zonotope U(X.center().zeros_like(),
                     Zonotope::generate_random(rng, 3, std::nullopt, 2).generators() * 0.2);
    const Tensor u = Zonotope::generate_random(rng, 3, std::nullopt, 1).center();
    return {LinearSys(sys.A(), eye),
            X,
            U,
            u,
            X.rand_point(kPoints, rng, "standard"),
            U.rand_point(kPoints, rng, "standard")};
}

/// Column j of a matrix of points.
Tensor column_of(const Tensor &P, int64_t j) { return P.slice(-1, j, j + 1); }

/// The state at the end of the step and inside it stays in the sets of every solution of a step.
void the_solutions_enclose_the_flow(const std::string &b) {
    cora::Rng rng(4);
    // every solution of a step is computed once per system, then checked at every sample
    for (int s = 0; s < kSystems; ++s) {
        const Case c = random_case(rng);
        const Tensor &A = c.sys.A();
        const HomogeneousSolution H = c.sys.homogeneous_solution(c.X, kStep, kTerms);
        const OneStep one = c.sys.one_step(c.X, c.U, c.u, kStep, kTerms);
        const AffineSolution affine = c.sys.affine_solution(c.X, c.u, kStep, kTerms);
        const Zonotope PU = c.sys.particular_solution_time_varying(c.U, kStep, kTerms);
        const ConstantSolution Pu = c.sys.particular_solution_constant(
            Zonotope(c.u, c.u.zeros_like({3, 0})), kStep, kTerms);
        // the first step of both algorithms, and the step after the first
        const ReachStep first = c.sys.init_reach(
            c.X, c.U, c.u, alg::Standard{.time_step = kStep, .taylor_terms = kTerms});
        const ReachStep wf = c.sys.init_reach(
            c.X, c.U, c.u, alg::WrappingFree{.time_step = kStep, .taylor_terms = kTerms});
        const ReachStep second =
            c.sys.post(first.state, alg::Standard{.time_step = kStep, .taylor_terms = kTerms});
        // each sample: a start point and a constant input of the set
        for (int64_t j = 0; j < kPoints; ++j) {
            const Tensor x0 = column_of(c.points, j), w = c.u + column_of(c.inputs, j);
            const Tensor homogeneous = exact_state(A, x0, A.zeros_like({3, 1}), kStep);
            check(H.time_point.contains(homogeneous), b + ": homogeneous end");
            check(Pu.time_point.contains(exact_state(A, A.zeros_like({3, 1}), c.u, kStep)),
                  b + ": constant input end");
            check(PU.contains(exact_state(A, A.zeros_like({3, 1}), column_of(c.inputs, j), kStep)),
                  b + ": time-varying input end");
            check(affine.time_point.contains(exact_state(A, x0, c.u, kStep)), b + ": affine end");
            // the end of the step and the states inside it, under the input u plus a sample
            const Tensor end = exact_state(A, x0, w, kStep);
            check(one.time_point.contains(end) && first.time_point.contains(end) &&
                      wf.time_point.contains(end),
                  b + ": end of the step");
            for (const double t : {0.0, 0.3 * kStep, 0.7 * kStep, kStep}) {
                const Tensor inside = exact_state(A, x0, w, t);
                check(one.time_interval.contains(inside) && first.time_interval.contains(inside) &&
                          wf.time_interval.contains(inside),
                      b + ": inside the step");
                check(H.time_interval.contains(exact_state(A, x0, A.zeros_like({3, 1}), t)),
                      b + ": homogeneous inside");
            }
            // the step after: from the same start and a constant input held over two steps
            check(second.time_point.contains(exact_state(A, x0, w, 2 * kStep)),
                  b + ": second step");
            check(second.time_interval.contains(exact_state(A, x0, w, 1.5 * kStep)),
                  b + ": inside the second step");
        }
    }
}

/// The adaptive error contains the input solution of the series it is cut from, at any order.
void the_adaptive_error_encloses_the_series(const std::string &b) {
    cora::Rng rng(4);
    for (int s = 0; s < kSystems; ++s) {
        const Case c = random_case(rng);
        for (const double zeta : {0.5, 1e-3, 1e-9}) {
            const Zonotope R = c.sys.error_solution_adaptive(c.U, std::nullopt, kStep, zeta);
            for (int64_t j = 0; j < kPoints; ++j)
                check(R.contains(
                          exact_state(c.sys.A(), c.u.zeros_like(), column_of(c.inputs, j), kStep)),
                      b + ": zeta " + std::to_string(zeta));
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_solutions_enclose_the_flow(b);
        the_adaptive_error_encloses_the_series(b);
    });
    return test::finish("linear_sys steps (property)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
