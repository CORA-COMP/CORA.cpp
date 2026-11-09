// test_nonlinear_sys_linearize - nonlinearSys linearize and the derivatives, by hand on f(x, u)
//
// f(x, u) = (x0 x1 + u, sin x0): its Jacobians, Hessians and third derivatives are written out,
// and the expansion point of linearize is the center moved by half a step of the flow.
//
// CORA: contDynamics/@nonlinearSys/linearize.m
// CORA: contDynamics/@nonlinearSys/nonlinearSys.m
// CORA: contDynamics/@contDynamics/derivatives.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
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

NonlinearSys aux_system() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[0] * x[1] + u[0], sin(x[0])};
        },
        2, 1);
}

/// The Jacobians and the Hessians at the point (1, 2), u = 0.5.
void the_derivatives_at_a_point(const std::string &b) {
    const NonlinearSys sys = aux_system();
    const Tensor x({1.0, 2.0}), u({0.5});
    check(test::close(sys.dynamics(x, u), {2.5, std::sin(1.0)}), b + ": f");
    const Jacobians J = sys.jacobian(x, u);
    check(test::close(J.A, {2.0, 1.0, std::cos(1.0), 0.0}) && test::close(J.B, {1.0, 0.0}),
          b + ": A and B");
    // the Hessians over z = (x0, x1, u)
    const std::vector<Tensor> H = sys.hessian(x, u);
    check(H.size() == 2 && test::close(H[0], {0, 1, 0, 1, 0, 0, 0, 0, 0}) &&
              test::close(H[1], {-std::sin(1.0), 0, 0, 0, 0, 0, 0, 0, 0}),
          b + ": the Hessians");
}

/// Over the box x0 in [0, 1], x1 in [1, 2], u in [0, 1]: d2f1/dx0^2 = -sin x0 in [-sin 1, 0],
/// d3f1/dx0^3 = -cos x0 in [-1, -cos 1], and no other third derivative.
void the_derivatives_over_a_box(const std::string &b) {
    const NonlinearSys sys = aux_system();
    const Range x(Tensor({0.0, 1.0}), Tensor({1.0, 2.0})), u(Tensor({0.0}), Tensor({1.0}));
    const std::vector<std::optional<Range>> H = sys.hessian_enclosure(x, u);
    check(H.size() == 18 && H[1] && H[3] && H[9] && !H[0] && !H[10], b + ": the nonzero Hessians");
    check(test::close(H[9]->lo, {-std::sin(1.0)}) && test::close(H[9]->hi, {0.0}),
          b + ": the range of -sin x0");
    const std::vector<std::optional<Range>> T = sys.third_order_enclosure(x, u);
    int nonzero = 0;
    for (const auto &t : T) nonzero += t.has_value();
    check(T.size() == 54 && nonzero == 1 && T[27], b + ": one third derivative");
    check(test::close(T[27]->lo, {-1.0}) && test::close(T[27]->hi, {-std::cos(1.0)}),
          b + ": the range of -cos x0");
}

/// x* = c + f(c, u*) dt/2 with u* the center of U; the input is B (U - u*), the constant f(x*, u*).
void linearize_by_hand(const std::string &b) {
    const NonlinearSys sys = aux_system();
    const Zonotope R(Tensor({1.0, 2.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const Zonotope U(Tensor({0.5}), Tensor({{0.2}}));
    const Linearized lin = sys.linearize(R, U, 0.1);
    const double x0 = 1 + 0.05 * 2.5, x1 = 2 + 0.05 * std::sin(1.0);
    check(test::close(lin.x_star, {x0, x1}) && test::close(lin.u_star, {0.5}), b + ": x* and u*");
    check(test::close(lin.f0, {x0 * x1 + 0.5, std::sin(x0)}) &&
              test::close(lin.u_const, {x0 * x1 + 0.5, std::sin(x0)}),
          b + ": f0 is the constant input");
    check(test::close(lin.sys.A(), {x1, x0, std::cos(x0), 0.0}), b + ": A at x*");
    check(test::close(lin.U.center(), {0.0, 0.0}) && test::close(lin.U.generators(), {0.2, 0.0}),
          b + ": the input set B (U - u*)");
}

// wrong arguments are described; the checks off give the same
void checks_and_their_switch() {
    const NonlinearSys sys = aux_system();
    const Zonotope R(Tensor({1.0, 2.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const Zonotope wrong(Tensor({1.0, 2.0, 3.0}), Tensor::eye(3));
    check(test::raises([&] { sys.linearize(wrong, std::nullopt, 0.1); }, Error::not_supported),
          "a set of another dimension");
    check(test::throws([&] { sys.linearize(R, R, 0.1); }), "an input set of another dimension");
    check(test::raises([&] { sys.linearize(R, std::nullopt, 0.0); }, Error::wrong_value),
          "time step 0");
    check(test::throws([&] { sys.jacobian(Tensor({1.0, 2.0})); }), "a point without its input");
    const Tensor on = sys.linearize(R, std::nullopt, 0.1).x_star;
    ChecksScope off(false);
    check(test::close(sys.linearize(R, std::nullopt, 0.1).x_star, on, 0), "the same, checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_derivatives_at_a_point(b);
        the_derivatives_over_a_box(b);
        linearize_by_hand(b);
    });
    checks_and_their_switch();
    return test::finish("nonlinearSys linearize");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
