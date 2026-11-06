// test_nonlinear_sys_simulate - nonlinearSys simulate: exact on a linear system, and its limits
//
// CORA: contDynamics/@nonlinearSys/simulate.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The harmonic oscillator x' = (x2, -x1) has the solution (cos t, -sin t) from (1, 0).
void follows_the_harmonic_oscillator(const std::string &b) {
    const NonlinearSys sys(
        [](const std::vector<Expr> &x) { return std::vector<Expr>{x[1], -x[0]}; }, 2);
    const std::vector<Tensor> x =
        sys.simulate({.initial_state = Tensor({{1.0, 0.0}, {0.0, 1.0}}), .t_final = 2.0},
                     alg::RungeKutta4{.time_step = 0.1});
    check(x.size() == 21, b + ": ceil(2.0 / 0.1) + 1 time points");
    double worst = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double t = 0.1 * double(k);
        const std::vector<double> d = x[k].data(); // rows (x1; x2), columns the two start points
        worst = std::max({worst, std::abs(d[0] - std::cos(t)), std::abs(d[2] + std::sin(t)),
                          std::abs(d[1] - std::sin(t)), std::abs(d[3] - std::cos(t))});
    }
    check(worst < 1e-9, b + ": a trajectory differs from the solution by " + std::to_string(worst));
}

/// The van der Pol oscillator settles on its limit cycle: x1 stays within about [-2.1, 2.1].
void van_der_pol_stays_on_its_limit_cycle(const std::string &b) {
    const NonlinearSys sys(
        [](const std::vector<Expr> &x) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0]};
        },
        2);
    const std::vector<Tensor> x =
        sys.simulate({.initial_state = Tensor({{1.4}, {2.3}}), .t_final = 20.0},
                     alg::RungeKutta4{.time_step = 0.05});
    double amplitude = 0;
    for (std::size_t k = x.size() / 2; k < x.size(); ++k)
        amplitude = std::max(amplitude, std::abs(x[k].data()[0]));
    check(amplitude > 1.9 && amplitude < 2.2,
          b + ": the amplitude is " + std::to_string(amplitude));
}

void wrong_arguments_are_described(const std::string &b) {
    const NonlinearSys sys(
        [](const std::vector<Expr> &x) { return std::vector<Expr>{x[1], -x[0]}; }, 2);
    check(test::throws([&] {
              sys.simulate({.initial_state = Tensor({1.0, 0.0, 1.0}), .t_final = 1.0},
                           alg::RungeKutta4{.time_step = 0.1});
          }),
          b + ": a start point of the wrong dimension");
    check(test::throws([&] {
              sys.simulate({.initial_state = Tensor({1.0, 0.0}), .t_final = 1.0},
                           alg::RungeKutta4{.time_step = 0.0});
          }),
          b + ": time step 0");
    check(test::throws([] { NonlinearSys([](const std::vector<Expr> &x) { return x; }, 0); }),
          "no states");
    check(test::throws([] {
              NonlinearSys([](const std::vector<Expr> &x) { return std::vector<Expr>{x[0]}; }, 2);
          }),
          "too few components");
}

/// An input is refused (no inputs); with the checks off the trajectory is the same.
void checks_and_their_switch() {
    const NonlinearSys sys = NonlinearSys(
        [](const std::vector<Expr> &x) {
            {
                return std::vector<Expr>{{x[1], -x[0]}};
            }
        },
        2);
    const Tensor x0({{1.0}, {0.0}});
    check(test::raises(
              [&] {
                  sys.simulate({.initial_state = x0, .input = x0, .t_final = 1},
                               alg::RungeKutta4{.time_step = 0.1});
              },
              Error::not_supported),
          "simulate: an input");
    const SimulationProblem problem{.initial_state = x0, .t_final = 1};
    const Tensor on = sys.simulate(problem, alg::RungeKutta4{.time_step = 0.1}).back();
    ChecksScope off(false);
    check(test::close(sys.simulate(problem, alg::RungeKutta4{.time_step = 0.1}).back(), on, 0),
          "simulate: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        follows_the_harmonic_oscillator(b);
        van_der_pol_stays_on_its_limit_cycle(b);
        wrong_arguments_are_described(b);
    });
    checks_and_their_switch();
    return test::finish("nonlinearSys simulate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
