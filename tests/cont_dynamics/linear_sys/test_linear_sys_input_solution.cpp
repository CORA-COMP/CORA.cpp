// test_linear_sys_input_solution - what an input set adds over one step, a public building block of
//
// reach
//
// CORA: contDynamics/@linearSys/particularSolution_constant.m
// CORA: contDynamics/@linearSys/particularSolution_timeVarying.m
// CORA authors: Mark Wetzlinger
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

/// U = {0}: nothing is added; a constant input u adds about time_step * B u.
void the_input_adds_its_solution() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}), Tensor({{1.0}, {0.0}}));
    const InputSolution zero =
        sys.input_solution(Zonotope(Tensor({0.0}), Tensor({{0.0}})), 0.01, 6);
    check(test::close(zero.Pu, std::vector<double>{0.0, 0.0}, 1e-15), "no constant input");
    const InputSolution one = sys.input_solution(Zonotope(Tensor({1.0}), Tensor({{0.0}})), 0.01, 6);
    check(test::close(one.Pu, std::vector<double>{0.01, 0.0}, 1e-4), "about time_step * B u");
}

/// No B, a set of the wrong dimension and a step of zero are refused; checks off, the same.
void checks_and_their_switch() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}), Tensor({{1.0}, {0.0}})),
                    autonomous = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}));
    const Zonotope U = Zonotope(Tensor({0.0}), Tensor({{0.1}}));
    check(test::raises([&] { autonomous.input_solution(U, 0.1, 4); }, Error::wrong_value), "no B");
    check(test::raises(
              [&] {
                  sys.input_solution(Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}})),
                                     0.1, 4);
              },
              Error::dimension_mismatch),
          "U of two dimensions");
    check(test::raises([&] { sys.input_solution(U, 0.0, 4); }, Error::wrong_value), "time_step 0");
    const Tensor on = sys.input_solution(U, 0.1, 4).PU.generators();
    ChecksScope off(false);
    check(test::close(sys.input_solution(U, 0.1, 4).PU.generators(), on, 0),
          "the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    the_input_adds_its_solution();
    checks_and_their_switch();
    return test::finish("linear_sys input_solution");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
