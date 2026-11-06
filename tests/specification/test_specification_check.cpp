// test_specification_check - specification check on any set and on a reachable set
//
// CORA: specification/@specification/check.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Any set answers, since a halfspace needs only the support function.
void any_contset(const std::string &b) {
    const Interval I(Tensor({1.0, 0.0}), Tensor({3.0, 2.0}));
    check(Specification::safe_set(Tensor({1.0, 0.0}), 3.0).check(I), b + ": an interval, safe");
    check(!Specification::safe_set(Tensor({1.0, 0.0}), 2.0).check(I), b + ": an interval, left");
    check(Specification::unsafe_set(Tensor({1.0, 0.0}), 0.5).check(I), b + ": an interval, clear");
    const ContSet &generic = I;
    check(Specification::safe_set(Tensor({0.0, 1.0}), 2.0).holds(generic).size() == 1,
          b + ": one answer for a single set");
}

/// x' = -x from [1, 2]: the set shrinks towards 0, but the first enclosure still reaches 2.
void on_a_reachable_set(const std::string &b) {
    const LinearSys sys(Tensor({{-1.0}}));
    const Zonotope X0(Tensor({1.5}), Tensor({{0.5}}));
    const ReachSet R = sys.reach({.initial_set = X0, .t_final = 1.0},
                                 alg::Standard{.time_step = 0.1, .taylor_terms = 8});

    check(Specification::safe_set(Tensor({1.0}), 2.1).check(R.time_interval),
          b + ": stays below 2.1");
    check(Specification::safe_set(Tensor({1.0}), 2.1).first_violation(R.time_interval) == -1,
          b + ": no violation");
    check(!Specification::safe_set(Tensor({1.0}), 1.9).check(R.time_interval), b + ": exceeds 1.9");
    check(Specification::safe_set(Tensor({1.0}), 1.9).first_violation(R.time_interval) == 0,
          b + ": exceeds 1.9 in the first step");

    // The lowest point decays as e^{-t}, passing 0.5 at t = 0.69; the first enclosure touching it.
    const Specification reaches_half = Specification::unsafe_set(Tensor({1.0}), 0.5);
    const int hit = reaches_half.first_violation(R.time_interval);
    check(hit >= 4 && hit <= 7, b + ": the flow reaches x <= 0.5 at step " + std::to_string(hit));
    check(reaches_half.check(
              std::vector<Zonotope>(R.time_interval.begin(), R.time_interval.begin() + hit)),
          b + ": clear before that step");
    check(Specification::unsafe_set(Tensor({1.0}), -1.0).check(R.time_interval),
          b + ": never below -1");
}

/// A halfspace of the wrong dimension is refused; with the checks off the answer is the same.
void checks_and_their_switch() {
    const Interval I(Tensor({1.0, 0.0}), Tensor({3.0, 2.0}));
    check(test::raises([&] { Specification::safe_set(Tensor({1.0, 0.0, 0.0}), 3.0).check(I); },
                       Error::dimension_mismatch),
          "check: a halfspace of three dimensions");
    const bool on = Specification::safe_set(Tensor({1.0, 0.0}), 3.0).check(I);
    ChecksScope off(false);
    check(Specification::safe_set(Tensor({1.0, 0.0}), 3.0).check(I) == on,
          "check: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        any_contset(b);
        on_a_reachable_set(b);
    });
    checks_and_their_switch();
    return test::finish("specification check");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
