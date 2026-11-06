// test_linear_sys_homogeneous_solution - homogeneous_solution: e^{A dt} X, the step and F X
//
// The end of the step, its enclosure and the curvature enlargement of one step.
//
// CORA: contDynamics/@linearSys/homogeneousSolution.m
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

/// For A = diag(-1, -2) the end of the step is diag(e^-dt, e^-2dt) X, known in closed form.
void the_end_of_the_step_is_e_adt_x(const std::string &b) {
    const HomogeneousSolution H =
        LinearSys(diagonal()).homogeneous_solution(initial_set(), kStep, kTerms);
    const double a = std::exp(-kStep), c = std::exp(-2 * kStep);
    check(test::close(H.time_point.center(), {a, 2 * c}, 1e-15), b + ": center");
    check(test::close(H.time_point.generators(), {a, 0.0, 0.0, 0.5 * c}, 1e-15),
          b + ": generators");
}

/// The enclosure of the step holds the start, the end and the trajectories between them.
void the_step_is_enclosed(const std::string &b) {
    const Tensor A = diagonal();
    const HomogeneousSolution H = LinearSys(A).homogeneous_solution(initial_set(), kStep, kTerms);
    const Zonotope X = initial_set();
    for (const std::vector<double> &d : {std::vector<double>{1, 0}, {0, 1}, {1, 1}, {-2, 1}}) {
        check(support(H.time_interval, d) >= support(X, d) - 1e-12, b + ": holds the start");
        check(support(H.time_interval, d) >= support(H.time_point, d) - 1e-12,
              b + ": holds the end");
    }
    // the corners of X flow along the curve between the two time points
    for (const std::vector<double> &x : {std::vector<double>{2, 2.5}, {0, 1.5}, {2, 1.5}, {0, 2.5}})
        for (const double t : {0.0, 0.03, 0.05, 0.08, kStep})
            check(H.time_interval.contains(flow(A, t, test::column(x))), b + ": a trajectory");
    // the curvature is F X
    const Interval F = LinearSys(A).correction_matrix_state(kStep, kTerms);
    check(test::close(H.state_curvature.interval().sup(), X.linear_map(F).interval().sup(), 1e-15),
          b + ": the curvature");
}

/// A dimension that does not fit, a step of zero and fewer than one term are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal());
    check(test::raises([&] { sys.homogeneous_solution(origin(3), kStep, kTerms); },
                       Error::dimension_mismatch),
          b + ": a set of three dimensions");
    check(test::raises([&] { sys.homogeneous_solution(initial_set(), 0.0, kTerms); },
                       Error::wrong_value),
          b + ": time_step 0");
    check(test::raises([&] { sys.homogeneous_solution(initial_set(), kStep, 0); },
                       Error::wrong_value),
          b + ": no terms");
    const Tensor on =
        sys.homogeneous_solution(initial_set(), kStep, kTerms).time_interval.generators();
    ChecksScope off(false);
    check(test::close(
              sys.homogeneous_solution(initial_set(), kStep, kTerms).time_interval.generators(), on,
              0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_end_of_the_step_is_e_adt_x(b);
        the_step_is_enclosed(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys homogeneous_solution");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
