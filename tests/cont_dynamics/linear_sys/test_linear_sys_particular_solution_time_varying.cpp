// test_linear_sys_particular_solution_time_varying - the input solution with the terms apart
//
// The terms of the series stay apart, so the set exceeds the image of U under the integral.
//
// CORA: contDynamics/@linearSys/particularSolution_timeVarying.m
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

/// U = [-1, 1] on the first state: every term of the series has the same sign there, so the support
/// is the sum of dt^(i+1)/(i+1)!, which is e^dt - 1 up to the truncated tail.
void the_support_is_the_sum_of_the_terms(const std::string &b) {
    const Zonotope U(test::column({0.0, 0.0}), Tensor({{1.0}, {0.0}}));
    const Zonotope PU = LinearSys(diagonal()).particular_solution_time_varying(U, kStep, kTerms);
    check(test::close(support(PU, {1, 0}), std::exp(kStep) - 1, 1e-13), b + ": e^dt - 1");
    check(test::close(support(PU, {-1, 0}), std::exp(kStep) - 1, 1e-13), b + ": symmetric");
    check(test::close(support(PU, {0, 1}), 0.0, 1e-13), b + ": nothing in the second state");
    // the first term is dt U: the generators come in the order of the series
    check(test::close(PU.generators().slice(-1, 0, 1), {kStep, 0.0}, 1e-15), b + ": dt U first");
    // dt U and one block per power of A, then the remainder: the center map and a box of 2 states
    check(PU.generators().shape().back() == (kTerms + 2) + 2, b + ": one block per term");
}

/// The center of U is part of the set: a point u gives the sum of the terms applied to u.
void the_center_enters(const std::string &b) {
    const Zonotope PU =
        LinearSys(diagonal()).particular_solution_time_varying(point({1.0, 1.0}), kStep, kTerms);
    // sum_i A^i dt^(i+1)/(i+1)! (1, 1) = ((1 - e^-dt) / 1, (1 - e^-2dt) / 2)
    check(test::close(PU.center(), {1 - std::exp(-kStep), (1 - std::exp(-2 * kStep)) / 2}, 1e-12),
          b + ": the center");
}

/// The origin adds nothing; A = 0 leaves dt U.
void nothing_in_nothing_out(const std::string &b) {
    const Zonotope PU =
        LinearSys(diagonal()).particular_solution_time_varying(origin(2), kStep, kTerms);
    check(PU.representsa("origin"), b + ": the origin stays");
    const Zonotope U(test::column({0.0, 0.0}), Tensor({{1.0}, {2.0}}));
    const Zonotope zero =
        LinearSys(Tensor::zeros({2, 2})).particular_solution_time_varying(U, kStep, 3);
    check(test::close(support(zero, {1, 1}), 3 * kStep, 1e-14), b + ": A = 0");
}

/// A set of the wrong dimension, a step of zero and fewer than one term are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal());
    check(test::raises([&] { sys.particular_solution_time_varying(origin(3), kStep, kTerms); },
                       Error::dimension_mismatch),
          b + ": a set of three dimensions");
    check(test::raises([&] { sys.particular_solution_time_varying(origin(2), -1.0, kTerms); },
                       Error::wrong_value),
          b + ": a negative step");
    const Zonotope U(test::column({0.0, 0.0}), Tensor({{1.0}, {0.0}}));
    const Tensor on = sys.particular_solution_time_varying(U, kStep, kTerms).generators();
    ChecksScope off(false);
    check(test::close(sys.particular_solution_time_varying(U, kStep, kTerms).generators(), on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_support_is_the_sum_of_the_terms(b);
        the_center_enters(b);
        nothing_in_nothing_out(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys particular_solution_time_varying");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
