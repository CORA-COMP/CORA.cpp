// test_linear_sys_nonlinear_sys - nonlinear_sys: x' = A x as a nonlinear system
//
// Inputs and an output map have no counterpart in NonlinearSys and are refused.
//
// CORA: contDynamics/@linearSys/nonlinearSys.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The dynamics of the converted system are A x, and its reach encloses that of the linear one.
void the_dynamics_are_a_x(const std::string &b) {
    const Tensor A({{-0.5, 1.0}, {-1.0, -0.5}});
    const NonlinearSys nl = LinearSys(A).nonlinear_sys();
    check(nl.dim() == 2, b + ": the dimension");
    const Tensor x = Tensor({{1.0, 0.0}, {2.0, -1.0}});
    check(test::close(nl.dynamics(x), A.matmul(x), 1e-15), b + ": f(x) = A x");
    // simulation of both agree
    const Tensor x0 = test::column({1.0, 2.0});
    const auto from_nonlinear =
        nl.simulate({.initial_state = x0, .t_final = 1.0}, alg::RungeKutta4{.time_step = 0.25});
    const auto from_linear =
        LinearSys(A).simulate({.initial_state = x0, .t_final = 1.0}, alg::Exact{.time_step = 0.25});
    check(test::close(from_nonlinear.back(), from_linear.back(), 1e-8),
          b + ": the same trajectory");
}

/// A system with inputs, or with an output matrix, does not convert; a batch neither.
void inputs_and_outputs_are_refused(const std::string &b) {
    check(test::raises([&] { LinearSys(diagonal(), Tensor::eye(2)).nonlinear_sys(); },
                       Error::not_supported),
          b + ": B");
    check(test::raises(
              [&] { LinearSys(diagonal(), Tensor::eye(2), Tensor({{1.0, 0.0}})).nonlinear_sys(); },
              Error::not_supported),
          b + ": B and C");
    const Tensor batch = Tensor::concat({diagonal().expand_dims(0), diagonal().expand_dims(0)}, 0);
    check(test::raises([&] { LinearSys(batch).nonlinear_sys(); }, Error::not_supported),
          b + ": a batch");
    ChecksScope off(false);
    check(LinearSys(diagonal()).nonlinear_sys().dim() == 2, b + ": with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_dynamics_are_a_x(b);
        inputs_and_outputs_are_refused(b);
    });
    return test::finish("linear_sys nonlinear_sys");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
