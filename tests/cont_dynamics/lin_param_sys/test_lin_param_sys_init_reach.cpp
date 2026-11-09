// test_lin_param_sys_init_reach - LinParamSys init_reach: the first step of reach, and its checks
//
// reach takes the center of the input set as the constant input and starts with init_reach, so
// its first sets are init_reach's. The first set over the step holds R_init (it starts there), and
// the mappings of the step are kept for post. Wrong dimensions are refused; the checks switch off.
//
// CORA: contDynamics/@linParamSys/initReach.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x' = A x + u + 0.5 e1 with A in [-1 2; -2 -1] +- 0.1 I.
LinParamSys aux_sys() {
    const MatZonotope A(Tensor({{-1.0, 2.0}, {-2.0, -1.0}}),
                        Tensor({{0.1, 0.0}, {0.0, 0.1}}).expand_dims(0));
    return LinParamSys(A, Tensor({{0.0}, {1.0}}), column({0.5, 0.0}));
}

/// The first step: reach's first sets, R_init inside the set over the step.
void first_step() {
    const LinParamSys sys = aux_sys();
    const Zonotope R0(column({1.0, 0.5}), Tensor({{0.05}, {0.0}}));
    const Zonotope U(column({0.1}), Tensor({{0.2}}));
    alg::LinParamStandard a;
    a.time_step = 0.05;
    const ReachSet R = sys.reach({R0, U, 0.1}, a);
    const LinParamStep first = sys.init_reach(R0, U - U.center(), U.center(), a);
    check(close(R.time_interval[0].center(), first.time_interval.center(), 0) &&
              close(R.time_interval[0].generators(), first.time_interval.generators(), 0) &&
              close(R.time_point[1].center(), first.time_point->center(), 0),
          "init_reach: the first sets of reach");
    check(first.time_interval.contains(R0, "exact", 1e-9), "init_reach: the step starts at R0");
    check(first.mappings.mapping.rows() == 2 && first.mappings.input.dim() == 2,
          "init_reach: the mappings of the step");
}

/// Wrong dimensions are refused; with the checks off the step is the same.
void checks_and_their_switch() {
    const LinParamSys sys = aux_sys();
    const Zonotope R0(column({1.0, 0.5}), Tensor({{0.05}, {0.0}}));
    const Zonotope U(column({0.0}), Tensor({{0.1}}));
    alg::LinParamStandard a;
    a.time_step = 0.05;
    check(test::raises([&] { sys.init_reach(R0, U, column({0.0, 0.0}), a); },
                       Error::dimension_mismatch),
          "init_reach: u_trans of the columns of B");
    check(
        test::raises(
            [&] { sys.init_reach(Zonotope(column({1.0}), Tensor({{1.0}})), U, column({0.0}), a); },
            Error::dimension_mismatch),
        "init_reach: R_init of the dimension of the system");
    const Tensor on = sys.init_reach(R0, U, column({0.0}), a).time_interval.center();
    ChecksScope off(false);
    check(close(sys.init_reach(R0, U, column({0.0}), a).time_interval.center(), on, 0),
          "init_reach: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    first_step();
    checks_and_their_switch();
    return test::finish("lin_param_sys init_reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
