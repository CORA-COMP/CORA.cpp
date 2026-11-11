// test_nonlin_das_sys_steady_state - nonlinDASys steadyState: Newton on the stacked system (f, g)
//
// x' = -x + y, 0 = y - x/2 - u at u = 0.5 has the steady state (1, 1): f = 0 and g = 0 there.
//
// CORA: contDynamics/@nonlinDASys/steadyState.m
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

/// The steady state of the linear system, and of a nonlinear one with a known root.
void the_steady_state_has_f_and_g_zero(const std::string &b) {
    const auto [x, y] = linear_system().steady_state(col(0.0), col(0.0), col(0.5));
    check(test::close(x, col(1.0), 1e-12) && test::close(y, col(1.0), 1e-12),
          b + ": the steady state is (1, 1)");
    // x' = y - 1, 0 = y - x^2 has the steady state (1, 1) and (-1, 1)
    const NonlinDASys square(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0] - 1}; },
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{y[0] - x[0] * x[0]}; }, 1, 0,
        1);
    const auto [xs, ys] = square.steady_state(col(2.0), col(2.0), Tensor::zeros({0, 1}));
    check(test::close(xs, col(1.0), 1e-9) && test::close(ys, col(1.0), 1e-9),
          b + ": a nonlinear steady state");
}

/// Wrong arguments are described, and the checks can be switched off.
void wrong_arguments_are_described(const std::string &b) {
    const NonlinDASys sys = linear_system();
    check(test::raises([&] { sys.steady_state(Tensor({1.0, 2.0}), col(0.0), col(0.5)); },
                       Error::dimension_mismatch),
          b + ": a state of the wrong size");
    const Tensor on = sys.steady_state(col(0.0), col(0.0), col(0.5)).first;
    ChecksScope off(false);
    check(test::close(sys.steady_state(col(0.0), col(0.0), col(0.5)).first, on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_steady_state_has_f_and_g_zero(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinDASys steadyState");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
