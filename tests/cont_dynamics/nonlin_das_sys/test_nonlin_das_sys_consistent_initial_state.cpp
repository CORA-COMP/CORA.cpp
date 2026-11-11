// test_nonlin_das_sys_consistent_initial_state - nonlinDASys consistentInitialState
//
// One step solves a constraint that is linear in y; y^2 = x converges quadratically. The last step
// is added (CORA's loop), the helper of priv_linearize stops before it. Wrong sizes are refused.
//
// CORA: contDynamics/@nonlinDASys/consistentInitialState.m
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

/// x' = y with the constraint g.
NonlinDASys with_constraint(const std::function<Expr(const Expr &x, const Expr &y)> &g) {
    return NonlinDASys(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; },
        [&g](const Vars &x, const Vars &y, const Vars &) { return Vars{g(x[0], y[0])}; }, 1, 0, 1);
}

/// Newton on y - x^2 is exact after one step; on y^2 - x it reaches sqrt(x).
void newton_solves_the_constraint(const std::string &b) {
    const Tensor no_input = Tensor::zeros({0, 1});
    const NonlinDASys square =
        with_constraint([](const Expr &x, const Expr &y) { return y - x * x; });
    check(test::close(square.consistent_initial_state(col(1.5), col(0.0), no_input), col(2.25),
                      1e-12),
          b + ": y = x^2 = 2.25");
    const NonlinDASys root =
        with_constraint([](const Expr &x, const Expr &y) { return y * y - x; });
    check(test::close(root.consistent_initial_state(col(2.0), col(1.0), no_input),
                      col(std::sqrt(2.0)), 1e-9),
          b + ": y = sqrt(2)");
}

/// Wrong arguments are described, and the checks can be switched off.
void wrong_arguments_are_described(const std::string &b) {
    const NonlinDASys sys = with_constraint([](const Expr &x, const Expr &y) { return y - x; });
    check(test::raises(
              [&] {
                  sys.consistent_initial_state(Tensor({1.0, 2.0}), col(0.0), Tensor::zeros({0, 1}));
              },
              Error::dimension_mismatch),
          b + ": a state of the wrong size");
    // a constraint without a root in y does not converge: g = y^2 + 1 has none
    const NonlinDASys none = with_constraint([](const Expr &, const Expr &y) { return y * y + 1; });
    check(test::raises(
              [&] { none.consistent_initial_state(col(0.0), col(1.0), Tensor::zeros({0, 1})); },
              Error::not_converged),
          b + ": no root");
    const Tensor on = sys.consistent_initial_state(col(2.0), col(0.0), Tensor::zeros({0, 1}));
    ChecksScope off(false);
    check(
        test::close(sys.consistent_initial_state(col(2.0), col(0.0), Tensor::zeros({0, 1})), on, 0),
        b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        newton_solves_the_constraint(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinDASys consistentInitialState");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
