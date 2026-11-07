// test_linear_sys_affine_solution - the homogeneous solution plus a constant input
//
// The end of the step adds the input, and the enclosure holds the trajectories.
//
// CORA: contDynamics/@linearSys/affineSolution.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x(dt) = e^{A dt} x0 + A^-1 (e^{A dt} - I) u: the center is that of the center, the set is e^{A
/// dt} X.
void the_end_of_the_step_adds_the_input(const std::string &b) {
    const Tensor A = diagonal(), u = test::column({1.0, -1.0});
    const AffineSolution S = LinearSys(A).affine_solution(initial_set(), u, kStep, kTerms);
    const Tensor center = flow(A, kStep, initial_set().center()) +
                          (A.inv().matmul((A * kStep).expm() - Tensor::eye(2))).matmul(u);
    check(test::close(S.time_point.center(), center, 1e-14), b + ": the center");
    check(test::close(S.time_point.generators(),
                      {std::exp(-kStep), 0.0, 0.0, 0.5 * std::exp(-2 * kStep)}, 1e-15),
          b + ": the generators of e^{A dt} X");
    check(test::close(S.constant.center(), {1 - std::exp(-kStep), -(1 - std::exp(-2 * kStep)) / 2},
                      1e-14),
          b + ": the part of the input");
}

/// The enclosure holds the start, the end and the trajectories of the corners between them.
void the_step_is_enclosed(const std::string &b) {
    const Tensor A = diagonal(), u = test::column({1.0, -1.0});
    const AffineSolution S = LinearSys(A).affine_solution(initial_set(), u, kStep, kTerms);
    const Tensor G = (A.inv().matmul((A * kStep).expm() - Tensor::eye(2)));
    for (const std::vector<double> &x :
         {std::vector<double>{2, 2.5}, {0, 1.5}, {2, 1.5}, {0, 2.5}}) {
        for (const double t : {0.0, 0.04, 0.07, kStep}) {
            // the state at t under the constant input u
            const Tensor xt = flow(A, t, test::column(x)) +
                              A.inv().matmul((A * t).expm() - Tensor::eye(2)).matmul(u);
            check(S.time_interval.contains(xt), b + ": a trajectory");
        }
    }
    check(S.time_interval.contains(S.time_point.center()), b + ": the end");
    check(test::close(G.matmul(u), S.constant.center(), 1e-14), b + ": the same constant part");
}

/// A u of the wrong shape, a set of the wrong dimension and a batch of systems are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal());
    const Tensor u = test::column({1.0, 0.0});
    check(test::raises(
              [&] { sys.affine_solution(initial_set(), test::column({1.0}), kStep, kTerms); },
              Error::dimension_mismatch),
          b + ": u of one row");
    check(test::raises([&] { sys.affine_solution(origin(3), u, kStep, kTerms); },
                       Error::dimension_mismatch),
          b + ": a set of three dimensions");
    check(
        test::raises([&] { sys.affine_solution(initial_set(), u, kStep, 0); }, Error::wrong_value),
        b + ": no terms");
    const Tensor on = sys.affine_solution(initial_set(), u, kStep, kTerms).time_point.center();
    ChecksScope off(false);
    check(test::close(sys.affine_solution(initial_set(), u, kStep, kTerms).time_point.center(), on,
                      0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_end_of_the_step_adds_the_input(b);
        the_step_is_enclosed(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys affine_solution");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
