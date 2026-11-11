// test_nonlin_das_sys_linearize - nonlinDASys linearize: the affine system with y eliminated
//
// x' = -x + y, 0 = y - x/2 - u: A_lin = A - C F^-1 D = -1/2, B_lin = B - C F^-1 E = 1, and the
// expansion point is half a step along the flow with y consistent. For x' = y, 0 = y - x^2 about
// x = 2: A_lin = 0 - 1 * (-4) = 4 and f0 = f - C F^-1 g = y - g = x^2 at the consistent point.
//
// CORA: contDynamics/@nonlinDASys/private/priv_linearize.m
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

/// The linear system x' = -x/2 + u after eliminating y.
void the_linear_system_is_reduced(const std::string &b) {
    const NonlinDASys sys(
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{-1 * x[0] + y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &u) { return Vars{y[0] - 0.5 * x[0] - u[0]}; },
        1, 1, 1);
    const Zonotope U(col(0.3), Tensor({{0.1}}));
    const DaeLinearized lin = sys.linearize(col(1.0), col(0.8), U, col(0.2), 0.1);
    // the point: u* = 0.3 + 0.2, x* = x0 + f(x0, y0, u*) dt/2 with f = -1 + 0.8, y* = x*/2 + u*
    const double x_star = 1.0 + (-1.0 + 0.8) * 0.05;
    check(test::close(lin.x_star, col(x_star), 1e-12), b + ": x* half a step on");
    check(test::close(lin.u_star, col(0.5), 1e-12), b + ": u* = center + constant input");
    check(test::close(lin.y_star, col(0.5 * x_star + 0.5), 1e-9), b + ": y* is consistent");
    check(test::close(lin.sys.A(), col(-0.5), 1e-12), b + ": A_lin = -1/2");
    // the input B_lin (U - center) = [-0.1, 0.1], and f0 = f at the point = -x* + y* = -x*/2 + u*
    check(test::close(lin.U.generators(), Tensor({{0.1}}), 1e-12) &&
              test::close(lin.U.center(), col(0.0), 1e-12),
          b + ": B_lin = 1 maps the centered input set");
    check(test::close(lin.u_const, col(-0.5 * x_star + 0.5), 1e-9), b + ": f0");
    check(test::close(lin.CF_inv, col(1.0), 1e-12) && test::close(lin.F_inv, col(1.0), 1e-12),
          b + ": C F^-1 = 1");
}

/// x' = y, 0 = y - x^2: A_lin = 4 at x = 2, from the sign of D and C F^-1.
void the_nonlinear_constraint_enters_the_flow(const std::string &b) {
    const NonlinDASys sys(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{y[0] - x[0] * x[0]}; }, 1, 0,
        1);
    // the step is so short that x* = x0 = 2 up to 1e-3, and y* = x*^2
    const DaeLinearized lin =
        sys.linearize(col(2.0), col(4.0), std::nullopt, Tensor::zeros({0, 1}), 1e-6);
    check(test::close(lin.sys.A(), col(4.0), 1e-3), b + ": A_lin = 4");
    check(test::close(lin.u_const, col(4.0), 1e-3), b + ": f0 = x^2");
}

/// Wrong arguments are described, and the checks can be switched off.
void wrong_arguments_are_described(const std::string &b) {
    const NonlinDASys sys(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{y[0] - x[0]}; }, 1, 0, 1);
    const Tensor none = Tensor::zeros({0, 1});
    check(test::raises([&] { sys.linearize(col(1.0), col(1.0), std::nullopt, none, 0.0); },
                       Error::wrong_value),
          b + ": time step 0");
    check(test::raises(
              [&] {
                  sys.linearize(col(1.0), col(1.0), Zonotope(col(0.0), Tensor({{1.0}})), none, 0.1);
              },
              Error::not_supported),
          b + ": an input set for a system without inputs");
    const Tensor on = sys.linearize(col(1.0), col(1.0), std::nullopt, none, 0.1).x_star;
    ChecksScope off(false);
    check(test::close(sys.linearize(col(1.0), col(1.0), std::nullopt, none, 0.1).x_star, on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_linear_system_is_reduced(b);
        the_nonlinear_constraint_enters_the_flow(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinDASys linearize");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
