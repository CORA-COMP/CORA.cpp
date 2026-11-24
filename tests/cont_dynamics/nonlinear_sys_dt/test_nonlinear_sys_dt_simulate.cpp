// test_nonlinear_sys_dt_simulate - nonlinearSysDT simulate: the recursion by hand, inputs, outputs
//
// x+ = (x0 + u, x1 + u cos x0) with y = x0 + x1: written out for two steps, a constant input and a
// signal, and the errors of a wrong signal or a final time that is not a multiple of the sampling
// time.
//
// CORA: contDynamics/@nonlinearSysDT/simulate.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys_dt/nonlinear_sys_dt.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

NonlinearSysDT aux_system() {
    return NonlinearSysDT(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[0] + u[0], x[1] + u[0] * cos(x[0])};
        },
        2, 1, 0.25,
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{x[0] + x[1]};
        },
        1);
}

void a_constant_input_and_a_signal(const std::string &b) {
    const NonlinearSysDT sys = aux_system();
    const Tensor x0({{0.0, 0.0}, {1.0, 2.0}});
    const DiscreteTrajectory T = sys.simulate(
        {.initial_state = x0, .input = test::vec({0.5}), .t_final = 0.5, .with_output = true});
    check(T.x.size() == 3 && T.y.size() == 3, b + ": three time points");
    // x(1) = (0.5, 0.5 cos 0) from the origin, and (1.5, 2 + 0.5 cos 1) from (1, 2)
    check(test::close(T.x[1], {0.5, 0.5, 1.5, 2 + 0.5 * std::cos(1.0)}), b + ": the first step");
    check(test::close(T.y[1], {1.0, 1.5 + 2 + 0.5 * std::cos(1.0)}), b + ": the outputs");
    // the signal 0.5 then 1 reads the second input at the second step
    const std::vector<Tensor> u = {test::vec({0.5}), test::vec({1.0})};
    const DiscreteTrajectory S =
        sys.simulate({.initial_state = x0, .input_signal = u, .t_final = 0.5});
    check(test::close(S.x[2],
                      {1.5, 0.5 + std::cos(0.5), 2.5, 2 + 0.5 * std::cos(1.0) + std::cos(1.5)}),
          b + ": a signal");
    check(S.y.empty(), b + ": no outputs unless asked");
}

/// The same trajectory with the checks off.
void the_same_with_the_checks_off(const std::string &b) {
    const NonlinearSysDT sys = aux_system();
    const DiscreteSimulationProblem problem{.initial_state = Tensor({{0.5, 0.5}}), .t_final = 0.5};
    const Tensor on = sys.simulate(problem).x.back();
    ChecksScope off(false);
    check(test::close(sys.simulate(problem).x.back(), on, 0), b + ": the same with the checks off");
}

void wrong_arguments_are_described(const std::string &b) {
    const NonlinearSysDT sys = aux_system();
    // a start point of two states
    const Tensor x0({{0.0, 0.0}});
    check(test::raises([&] { sys.simulate({.initial_state = x0, .t_final = 0.3}); },
                       Error::wrong_value),
          b + ": a final time that is no multiple");
    check(test::raises(
              [&] {
                  sys.simulate({.initial_state = x0,
                                .input_signal = std::vector<Tensor>{test::vec({0.5})},
                                .t_final = 0.5});
              },
              Error::dimension_mismatch),
          b + ": a signal of the wrong length");
    check(test::raises([&] { sys.simulate({.initial_state = Tensor({{0.0}}), .t_final = 0.5}); },
                       Error::dimension_mismatch),
          b + ": a start point of the wrong dimension");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_constant_input_and_a_signal(b);
        wrong_arguments_are_described(b);
        the_same_with_the_checks_off(b);
    });
    return test::finish("nonlinearSysDT simulate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
