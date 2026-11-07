// test_linear_sys_particular_solution_constant - the solution of a constant input
//
// The closed form for an invertible A, the truncated series with remainder for a singular one.
//
// CORA: contDynamics/@linearSys/particularSolution_constant.m
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

/// A^-1 (e^{A dt} - I) u = ((1 - e^-dt) / 1, (1 - e^-2dt) / 2) for u = (1, 1), without a remainder.
void an_invertible_a_integrates_in_closed_form(const std::string &b) {
    const ConstantSolution P =
        LinearSys(diagonal()).particular_solution_constant(point({1.0, 1.0}), kStep, kTerms);
    check(test::close(P.time_point.center(), {1 - std::exp(-kStep), (1 - std::exp(-2 * kStep)) / 2},
                      1e-14),
          b + ": the end of the step");
    check(P.time_point.generators().shape().back() == 0, b + ": exact, no remainder");
    // the enclosure holds the origin, the end of the step and the curvature of the input
    check(P.time_interval.contains(test::column({0.0, 0.0})), b + ": the origin");
    check(P.time_interval.contains(P.time_point.center()), b + ": the end");
}

/// The double integrator is singular: x' = x2 + 0, with x2' = u, the series ends after A^1.
void a_singular_a_sums_the_series(const std::string &b) {
    const ConstantSolution P =
        LinearSys(integrator()).particular_solution_constant(point({0.0, 1.0}), kStep, kTerms);
    // (dt I + A dt^2/2) (0, 1)
    check(test::close(P.time_point.center(), {kStep * kStep / 2, kStep}, 1e-15),
          b + ": the series");
    // the curvature G u is a box below zero in the first state, where the velocity curves the path
    check(support(P.input_curvature, {-1, 0}) > 1e-4 &&
              std::abs(support(P.input_curvature, {0, 1})) < 1e-15,
          b + ": the curvature");
    check(P.time_interval.contains(test::column({0.0, 0.0})) &&
              P.time_interval.contains(P.time_point.center()),
          b + ": the enclosure");
}

/// The origin adds nothing; a set U is mapped as a whole.
void the_origin_and_a_set(const std::string &b) {
    const LinearSys sys(diagonal());
    const ConstantSolution zero = sys.particular_solution_constant(origin(2), kStep, kTerms);
    check(zero.time_point.representsa("origin") && zero.input_curvature.representsa("origin") &&
              zero.time_interval.representsa("origin"),
          b + ": the origin stays");
    const Zonotope U(test::column({0.0, 0.0}), Tensor({{1.0}, {0.0}}));
    const ConstantSolution P = sys.particular_solution_constant(U, kStep, kTerms);
    check(test::close(support(P.time_point, {1, 0}), 1 - std::exp(-kStep), 1e-14),
          b + ": a set is mapped by the integral");
}

/// A set of the wrong dimension, a step of zero and a batch of systems are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal());
    // the dimension, the step and the batch are refused
    check(test::raises([&] { sys.particular_solution_constant(origin(3), kStep, kTerms); },
                       Error::dimension_mismatch),
          b + ": a set of three dimensions");
    check(test::raises([&] { sys.particular_solution_constant(origin(2), 0.0, kTerms); },
                       Error::wrong_value),
          b + ": time_step 0");
    const Tensor batch =
        Tensor::concat({diagonal().expand_dims(0), integrator().expand_dims(0)}, 0);
    check(test::raises(
              [&] { LinearSys(batch).particular_solution_constant(origin(2), kStep, kTerms); },
              Error::not_supported),
          b + ": a batch of systems");
    // the checks off give the same end of the step
    const Tensor on =
        sys.particular_solution_constant(point({1.0, 1.0}), kStep, kTerms).time_point.center();
    ChecksScope off(false);
    check(
        test::close(
            sys.particular_solution_constant(point({1.0, 1.0}), kStep, kTerms).time_point.center(),
            on, 0),
        b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        an_invertible_a_integrates_in_closed_form(b);
        a_singular_a_sums_the_series(b);
        the_origin_and_a_set(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys particular_solution_constant");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
