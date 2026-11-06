// test_linear_sys_validate - the checks of a reach problem against the system, also for algorithms
//
// of Python
//
// CORA: contDynamics/@contDynamics/contDynamics.m
// CORA: contDynamics/@linearSys/linearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Every part of the problem is checked: the sets, their dimensions, B and t_final.
void every_part_is_checked() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}), Tensor({{1.0}, {0.0}})),
                    autonomous = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}));
    const Zonotope X0 = Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}})),
                   U = Zonotope(Tensor({0.0}), Tensor({{0.1}}));
    check(test::raises([&] { sys.validate({.t_final = 1}); }, Error::wrong_value),
          "no initial set");
    check(test::raises(
              [&] {
                  sys.validate(
                      {.initial_set = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                               Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}})),
                       .t_final = 1});
              },
              Error::dimension_mismatch),
          "an initial set of three dimensions");
    check(
        test::raises([&] { sys.validate({.initial_set = X0, .t_final = -1}); }, Error::wrong_value),
        "t_final -1");
    check(test::raises(
              [&] { autonomous.validate({.initial_set = X0, .input_set = U, .t_final = 1}); },
              Error::wrong_value),
          "an input set without B");
    check(test::raises([&] { sys.validate({.initial_set = X0, .input_set = X0, .t_final = 1}); },
                       Error::dimension_mismatch),
          "an input set of two dimensions");
    sys.validate({.initial_set = X0, .input_set = U, .t_final = 1});
}

/// With the checks off validate passes anything.
void checks_and_their_switch() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}));
    ChecksScope off(false);
    check(!test::throws([&] { sys.validate({.t_final = -1}); }), "nothing is checked");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    every_part_is_checked();
    checks_and_their_switch();
    return test::finish("linear_sys validate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
