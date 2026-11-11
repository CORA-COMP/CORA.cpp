// test_nonlin_das_sys_simulate - nonlinDASys simulate: Runge-Kutta with y solved at every stage
//
// The linear system x' = -x + y, 0 = y - x/2 - u is x' = -x/2 + u with y = x/2 + u: the trajectory
// is the closed-form solution. Wrong arguments are refused.
//
// CORA: contDynamics/@nonlinDASys/simulate.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vars = std::vector<Expr>;

Tensor col(double v) { return Tensor({v}); }

NonlinDASys linear_system() {
    return NonlinDASys(
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{-1 * x[0] + y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &u) { return Vars{y[0] - 0.5 * x[0] - u[0]}; },
        1, 1, 1);
}

/// x(1) = e^{-1/2} + 2 u (1 - e^{-1/2}), and y = x/2 + u at every sample.
void the_trajectory_is_the_closed_form(const std::string &b) {
    const DaeTrajectory T = linear_system().simulate(col(1.0), col(0.0), col(0.2), 1.0,
                                                     alg::RungeKutta4{.time_step = 0.1});
    const double exact = std::exp(-0.5) + 0.4 * (1 - std::exp(-0.5));
    check(T.x.shape() == std::vector<int64_t>({1, 11}) && T.y.shape() == T.x.shape(),
          b + ": 10 steps, 11 samples");
    check(test::close(T.x.slice(-1, 10, 11), col(exact), 1e-6), b + ": x(1) is exact");
    check(test::close(T.y.slice(-1, 10, 11), col(0.5 * exact + 0.2), 1e-6), b + ": y = x/2 + u");
}

/// Wrong arguments are described, and the checks can be switched off.
void wrong_arguments_are_described(const std::string &b) {
    const NonlinDASys sys = linear_system();
    const alg::RungeKutta4 rk{.time_step = 0.1};
    check(test::raises([&] { sys.simulate(Tensor({1.0, 2.0}), col(0.0), col(0.2), 1.0, rk); },
                       Error::dimension_mismatch),
          b + ": a state of the wrong size");
    check(test::raises([&] { sys.simulate(col(1.0), col(0.0), col(0.2), 0.0, rk); },
                       Error::wrong_value),
          b + ": a horizon of 0");
    const Tensor on = sys.simulate(col(1.0), col(0.0), col(0.2), 0.5, rk).x;
    ChecksScope off(false);
    check(test::close(sys.simulate(col(1.0), col(0.0), col(0.2), 0.5, rk).x, on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_trajectory_is_the_closed_form(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinDASys simulate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
