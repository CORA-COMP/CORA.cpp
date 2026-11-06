// test_linear_sys_error_solution_adaptive - the series of the input solution, cut adaptively
//
// The series is cut by the gain of a term or at a given order.
//
// CORA: contDynamics/@linearSys/errorSolution_adaptive.m
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

Zonotope dynamic_part() { return {test::column({0.0, 0.0}), Tensor({{1.0}, {0.5}})}; }

/// At a given order the set is the series of the time-varying input solution up to that order.
void a_given_order_is_the_series(const std::string &b) {
    const LinearSys sys(diagonal());
    for (const int terms : {2, 5, 8}) {
        const Zonotope R =
            sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 1e-3, terms);
        const Zonotope PU = sys.particular_solution_time_varying(dynamic_part(), kStep, terms);
        for (const std::vector<double> &d : {std::vector<double>{1, 0}, {0, 1}, {1, -2}})
            check(test::close(support(R, d), support(PU, d), 1e-13),
                  b + ": order " + std::to_string(terms));
    }
}

/// The gain stops the series where a term adds less than zeta_tabs; a smaller zeta goes further.
void the_gain_decides_where_to_stop(const std::string &b) {
    const LinearSys sys(diagonal());
    const Zonotope coarse = sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 0.1);
    const Zonotope fine = sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 1e-9);
    check(coarse.generators().shape().back() < fine.generators().shape().back(),
          b + ": a smaller zeta keeps more terms");
    // both enclose the exact solution int_0^dt e^{A s} ds Vdyn, whose support along e1 is 1 - e^-dt
    check(support(coarse, {1, 0}) >= 1 - std::exp(-kStep) - 1e-13 &&
              support(fine, {1, 0}) >= 1 - std::exp(-kStep) - 1e-13,
          b + ": the exact solution is inside");
    check(support(fine, {1, 0}) <= std::exp(kStep) - 1 + 1e-10, b + ": and not far outside");
}

/// The static part adds e^{A dt}'s integral applied to Vstat.
void the_static_part_is_added(const std::string &b) {
    const LinearSys sys(diagonal());
    const Zonotope Vstat = point({1.0, 1.0});
    const Zonotope with = sys.error_solution_adaptive(dynamic_part(), Vstat, kStep, 1e-3, 10);
    const Zonotope without =
        sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 1e-3, 10);
    check(test::close(with.center(),
                      without.center() +
                          test::column({1 - std::exp(-kStep), (1 - std::exp(-2 * kStep)) / 2}),
                      1e-12),
          b + ": the center moves by the integral of the static part");
}

/// A bad zeta, an order below two, a set of the wrong dimension and a batch of systems are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal());
    // zeta and the order are ranges; the sets are checked against the dimension of A
    check(
        test::raises([&] { sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 1.5); },
                     Error::wrong_value),
        b + ": zeta 1.5");
    check(test::raises(
              [&] { sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 0.1, 1); },
              Error::wrong_value),
          b + ": order 1");
    check(test::raises([&] { sys.error_solution_adaptive(origin(3), std::nullopt, kStep, 0.1); },
                       Error::dimension_mismatch),
          b + ": a set of three dimensions");
    check(test::raises([&] { sys.error_solution_adaptive(dynamic_part(), origin(3), kStep, 0.1); },
                       Error::dimension_mismatch),
          b + ": Vstat of three dimensions");
    // the checks off give the same set
    const Tensor on =
        sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 0.1).generators();
    ChecksScope off(false);
    check(test::close(
              sys.error_solution_adaptive(dynamic_part(), std::nullopt, kStep, 0.1).generators(),
              on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_given_order_is_the_series(b);
        the_gain_decides_where_to_stop(b);
        the_static_part_is_added(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys error_solution_adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
