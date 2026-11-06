// test_linear_sys_init_reach - the first step of the algorithms standard and wrapping_free
//
// The first step is that of reach; an algorithm without such a step is refused.
//
// CORA: contDynamics/@linearSys/initReach.m
// CORA authors: Mark Wetzlinger, Matthias Althoff
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

const alg::Standard kStandard{.time_step = kStep, .taylor_terms = kTerms};
const alg::WrappingFree kWrappingFree{.time_step = kStep, .taylor_terms = kTerms};

/// Without an input the first step is the enclosure of reach: the same boxes.
void the_first_step_is_that_of_reach(const std::string &b) {
    const LinearSys sys(diagonal());
    const ReachSet R = sys.reach({.initial_set = initial_set(), .t_final = kStep}, kStandard);
    for (const alg::LinearReach &a :
         {alg::LinearReach(kStandard), alg::LinearReach(kWrappingFree)}) {
        const ReachStep s = sys.init_reach(initial_set(), std::nullopt, std::nullopt, a);
        const Interval got = s.time_interval.interval(), want = R.time_interval[0].interval();
        check(test::close(got.inf(), want.inf(), 1e-13) &&
                  test::close(got.sup(), want.sup(), 1e-13),
              b + ": the box of the step");
        check(test::close(s.time_point.center(), R.time_point[1].center(), 1e-14),
              b + ": the end of the step");
    }
}

/// An algorithm that is no step of init_reach, an input without B and a bad set are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal(), Tensor::eye(2));
    const Zonotope U = origin(2);
    // from_start and decomposition have no first step, nor does a system without inputs
    check(test::raises(
              [&] {
                  sys.init_reach(initial_set(), U, std::nullopt,
                                 alg::FromStart{.time_step = kStep});
              },
              Error::not_supported),
          b + ": from_start has no init_reach");
    check(test::raises(
              [&] { LinearSys(diagonal()).init_reach(initial_set(), U, std::nullopt, kStandard); },
              Error::wrong_value),
          b + ": an input needs B");
    // a set of the wrong dimension and an algorithm without a time step
    check(test::raises([&] { sys.init_reach(origin(3), std::nullopt, std::nullopt, kStandard); },
                       Error::dimension_mismatch),
          b + ": a set of three dimensions");
    check(test::raises(
              [&] { sys.init_reach(initial_set(), std::nullopt, std::nullopt, alg::Standard{}); },
              Error::wrong_value),
          b + ": no time_step");
    // the checks off give the same first step
    const Tensor on =
        sys.init_reach(initial_set(), U, std::nullopt, kStandard).time_interval.center();
    ChecksScope off(false);
    check(test::close(
              sys.init_reach(initial_set(), U, std::nullopt, kStandard).time_interval.center(), on,
              0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_first_step_is_that_of_reach(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys init_reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
