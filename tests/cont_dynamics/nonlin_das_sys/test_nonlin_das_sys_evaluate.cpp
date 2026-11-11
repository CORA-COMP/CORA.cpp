// test_nonlin_das_sys_evaluate - nonlinDASys: the equations and their derivatives
//
// x' = y, 0 = y - x^2 (index 1, F = 1): the values and the Jacobian blocks at a point, the Hessians
// over a box (zero entries empty), and the checks of a wrong number of equations.
//
// CORA: contDynamics/@nonlinDASys/nonlinDASys.m
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

/// x' = y, 0 = y - x^2: the algebraic variable is x^2.
NonlinDASys square_system() {
    return NonlinDASys(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{y[0] - x[0] * x[0]}; }, 1, 0,
        1);
}

Tensor col(double v) { return Tensor({v}); }

/// f and g at columns of points, and the blocks of the Jacobian at one.
void the_values_and_the_jacobian(const std::string &b) {
    const NonlinDASys sys = square_system();
    const Tensor x = Tensor({{1.0, 2.0}}), y = Tensor({{3.0, 3.0}}), u = Tensor::zeros({0, 2});
    check(test::close(sys.dynamics(x, y, u), Tensor({{3.0, 3.0}})), b + ": f = y");
    check(test::close(sys.constraints(x, y, u), Tensor({{2.0, -1.0}})), b + ": g = y - x^2");
    const DaeJacobians J = sys.jacobian(col(2.0), col(3.0), Tensor::zeros({0, 1}));
    check(test::close(J.A, col(0.0)) && test::close(J.C, col(1.0)), b + ": A = 0, C = 1");
    check(test::close(J.D, col(-4.0)) && test::close(J.F, col(1.0)), b + ": D = -2 x, F = 1");
    check(J.B.shape() == std::vector<int64_t>({1, 0}) &&
              J.E.shape() == std::vector<int64_t>({1, 0}),
          b + ": no input columns");
}

/// The Hessian of g over a box is -2 in the (x, x) entry, the zeros are empty.
void the_hessian_over_a_box(const std::string &b) {
    const NonlinDASys sys = square_system();
    const Range x(col(0.0), col(1.0)), y(col(0.0), col(1.0)), u(Tensor::zeros({0, 1}));
    const DaeHessians H = sys.hessian_enclosure(x, y, u);
    // z = (x, y): row g, entry (x, x) = -2; every other entry and every entry of f is empty
    check(H.f.size() == 4 && H.g.size() == 4, b + ": one row of d x d = 4 entries each");
    for (const auto &h : H.f) check(!h, b + ": f is linear");
    check(H.g[0] && test::close(H.g[0]->lo, col(-2.0)) && test::close(H.g[0]->hi, col(-2.0)),
          b + ": d2g/dx2 = -2");
    check(!H.g[1] && !H.g[2] && !H.g[3], b + ": the other entries of g are 0");
}

/// Wrong arguments are described: a mismatch of sizes, a wrong number of equations.
void wrong_arguments_are_described(const std::string &b) {
    const NonlinDASys sys = square_system();
    check(
        test::raises([&] { sys.constraints(Tensor({1.0, 2.0}), col(0.0), Tensor::zeros({0, 1})); },
                     Error::dimension_mismatch),
        b + ": a state of the wrong size");
    check(test::raises(
              [&] {
                  NonlinDASys(
                      [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0], y[0]}; },
                      [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; }, 1, 0,
                      1);
              },
              Error::wrong_input_in_constructor),
          b + ": f with two rows for one state");
    check(test::raises([&] { NonlinDASys(nullptr, nullptr, 0, 0, 1); },
                       Error::wrong_input_in_constructor),
          b + ": no state");
    // with the checks off the values are the same
    const Tensor on = sys.constraints(col(2.0), col(3.0), Tensor::zeros({0, 1}));
    ChecksScope off(false);
    check(test::close(sys.constraints(col(2.0), col(3.0), Tensor::zeros({0, 1})), on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_values_and_the_jacobian(b);
        the_hessian_over_a_box(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinDASys evaluate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
