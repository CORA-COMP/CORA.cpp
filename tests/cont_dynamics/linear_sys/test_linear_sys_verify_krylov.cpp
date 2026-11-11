// test_linear_sys_verify_krylov - linear_sys verify with alg::ReachAvoidKrylov (properties)
//
// The algorithm has no counterpart in MATLAB CORA v2026.1.0 (the competition's copy has its own
// reachSuppFuncsKrylov) and none in CORALean, so there are no reference numbers: the verdicts are
// those of the exact support functions, which this system has in closed form. A = [-0.1 1; -1 -0.1]
// rotates and decays: e^{At} = e^{-0.1 t} [cos t, sin t; -sin t, cos t]; with R0 = [1;0] +
// 0.1 [-1, 1]^2 the support function along e_1 is e^{-0.1 t} (cos t + 0.1 (|cos t| + |sin t|)),
// at most 1.1 (t = 0); along e_2 it is e^{-0.1 t} (-sin t + 0.1 (|cos t| + |sin t|)) <= 0.1 *
// sqrt(2) + 0 near t = 0 and its maximum is 0.62 (t about 4.8).
//
// CORA.cpp only: no counterpart in CORA v2026.1.0 (see priv_verify_ra_krylov.cpp)
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/sparse_matrix.h"
#include "spec_testing.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Tensor mat(int rows, int cols, const std::vector<double> &row_major) {
    return Tensor::from_data(row_major, {rows, cols});
}

Zonotope zono(const std::vector<double> &c, const std::vector<double> &G, int m) {
    const int n = static_cast<int>(c.size());
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, m})};
}

/// The specification a'y <= b, safe or unsafe.
Specification safe(const std::vector<double> &a, double b) {
    return Specification::safe_set(test::halfspace(test::column(a), b));
}
Specification unsafe(const std::vector<double> &a, double b) {
    return Specification::unsafe_set(test::halfspace(test::column(a), b));
}

/// The unsafe box [lo1, hi1] x [lo2, hi2] as four halfspaces.
Specification unsafe_box(double lo1, double hi1, double lo2, double hi2) {
    return Specification::unsafe_set(test::polytope({{test::column({1, 0}), hi1},
                                                     {test::column({0, 1}), hi2},
                                                     {test::column({-1, 0}), -lo1},
                                                     {test::column({0, -1}), -lo2}}));
}

const std::vector<double> Aosc{-0.1, 1, -1, -0.1};

/// The maximum over t in [0, t_final] of the support function of y = C x along l, x(0) in R0,
/// u in U constant in time, on a grid of n points (z(t) = e^{A' t} C' l, integrals by the
/// trapezoid).
double exact_maximum(const Tensor &A, const Tensor &B, const Tensor &C, const Zonotope &R0,
                     const Zonotope &U, const Tensor &l, double t_final, int n) {
    const Tensor d = C.matrix_transpose().matmul(l);
    const Tensor At = A.matrix_transpose();
    const Tensor &G0 = R0.generators(), &GU = U.generators();
    double best = -1e300, integral = 0, previous_f = 0;
    for (int i = 0; i <= n; ++i) {
        const double t = t_final * i / n;
        const Tensor z = (At * t).expm().matmul(d);
        const Tensor zb = B.matrix_transpose().matmul(z);
        const double f = zb.matrix_transpose().matmul(U.center()).item() +
                         zb.matrix_transpose().matmul(GU).abs().sum().item();
        if (i > 0) integral += 0.5 * (f + previous_f) * t_final / n;
        previous_f = f;
        const double h = z.matrix_transpose().matmul(R0.center()).item() +
                         z.matrix_transpose().matmul(G0).abs().sum().item() + integral;
        best = std::max(best, h);
    }
    return best;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const Zonotope R0 = zono({1, 0}, {0.1, 0, 0, 0.1}, 2);
    const Zonotope none = zono({0}, {}, 0);
    const Zonotope u05 = zono({0.05}, {0.05}, 1);
    const LinearSys osc0(mat(2, 2, Aosc), mat(2, 1, {0, 0}));
    const LinearSys osc1(mat(2, 2, Aosc), mat(2, 1, {0, 1}));
    const alg::ReachAvoidKrylov K{};
    // a violated specification halves the step until min_time_step: a coarse one ends it early
    const alg::ReachAvoidKrylov coarse{.min_time_step = 1e-3};
    auto verified = [&](const LinearSys &sys, const Zonotope &U, const Specification &spec) {
        return sys.verify({R0, U, 5}, {spec}, coarse).verified;
    };

    // Safe sets: the maximum along e_1 is 1.1 at t = 0.
    check(verified(osc0, none, safe({1, 0}, 1.3)), "safe y1 <= 1.3 holds");
    check(!verified(osc0, none, safe({1, 0}, 1.0)), "safe y1 <= 1.0 is hit by R0");
    check(verified(osc0, none, safe({0, 1}, 0.7)), "safe y2 <= 0.7 holds");
    check(!verified(osc0, none, safe({0, 1}, 0.5)), "safe y2 <= 0.5 is hit at t about 4.8");
    // Unsafe halfspaces {a'y <= b}: y1 >= 1.5 is avoided, y1 >= 1.0 is hit by R0.
    check(verified(osc0, none, unsafe({-1, 0}, -1.5)), "unsafe y1 >= 1.5 is avoided");
    check(!verified(osc0, none, unsafe({-1, 0}, -1.0)), "unsafe y1 >= 1.0 is hit");
    // Unsafe boxes: a row separates the box from the reachable set on every step.
    check(verified(osc0, none, unsafe_box(1.5, 2, -1, 1)), "unsafe box right of the set");
    check(verified(osc0, none, unsafe_box(-1, 1, 1.5, 2)), "unsafe box above the set");
    check(!verified(osc0, none, unsafe_box(0.5, 2, -1, 1)), "unsafe box on the set");
    // An input set adds to the support: u in [0, 0.1] moves y2 by at most 0.1 per unit time.
    check(verified(osc1, u05, safe({1, 0}, 2)), "input: safe y1 <= 2 holds");
    check(!verified(osc1, u05, safe({0, 1}, 0.5)), "input: safe y2 <= 0.5 is hit");
    // The verdict of the support function algorithm, where it applies.
    for (const Specification &spec : {safe({1, 0}, 1.3), safe({0, 1}, 0.7), safe({0, 1}, 0.5),
                                      unsafe({-1, 0}, -1.5), unsafe({-1, 0}, -1.0)}) {
        check(osc1.verify({R0, u05, 5}, {spec}, coarse).verified ==
                  osc1.verify({R0, u05, 5}, {spec}, alg::ReachAvoidSupportFunction{}).verified,
              "the same verdict as the support function algorithm");
    }
    // A sparse state matrix gives the same verdict, and the step is the first of t_final / 1000.
    const LinearSys sparse(SparseMatrix(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {-0.1, 1, -1, -0.1}),
                           mat(2, 1, {0, 1}));
    const VerifyResult dense = osc1.verify({R0, u05, 5}, {safe({1, 0}, 2)}, coarse);
    const VerifyResult sp = sparse.verify({R0, u05, 5}, {safe({1, 0}, 2)}, coarse);
    check(sp.verified == dense.verified && sp.num_steps == dense.num_steps,
          "a sparse A gives the same verdict and steps");
    check(dense.num_steps == 1000 && test::close(dense.time_step, 0.005, 1e-15),
          "the first step is t_final / 1000");
    // The step is halved while the bound is too wide: a margin of 0.001 needs more than one pass.
    const VerifyResult tight = osc0.verify({R0, none, 5}, {safe({1, 0}, 1.1001)}, K);
    check(tight.verified && tight.iterations > 1, "a tight margin halves the step");
    // A 4-state system with two inputs: 0.2 percent above the exact maximum verifies, below not.
    const Tensor A4 =
        mat(4, 4,
            {-0.3, 2.0, 0.1, 0.0, -2.0, -0.3, 0.0, 0.2, 0.0, 0.1, -0.6, 1.1, 0.3, 0.0, -1.1, -0.6});
    const Tensor B4 = mat(4, 2, {1, 0, 0, 0.5, -0.5, 1, 0.2, 0.2});
    const Tensor C4 = mat(2, 4, {1, 0.5, 0, 0.2, 0, -0.3, 1, 0.4});
    const Zonotope R4 = zono({1, -1, 0.5, 2}, {0.2, 0, 0, 0.1, 0, 0.1, 0.3, 0}, 2);
    const Zonotope U4 = zono({0.1, -0.05}, {0.05, 0, 0, 0.02}, 2);
    const LinearSys sys4(A4, B4, C4);
    for (const std::vector<double> &l : {std::vector<double>{1, 0}, {0, 1}, {-0.6, 0.8}}) {
        const double h = exact_maximum(A4, B4, C4, R4, U4, test::column(l), 4, 4000);
        const std::vector<Specification> above{safe(l, h + 0.002 * std::abs(h) + 1e-3)};
        const std::vector<Specification> below{safe(l, h - 0.002 * std::abs(h) - 1e-3)};
        check(sys4.verify({R4, U4, 4}, above, K).verified, "exact maximum: 0.2 percent above");
        check(
            !sys4.verify({R4, U4, 4}, below, alg::ReachAvoidKrylov{.min_time_step = 1e-4}).verified,
            "exact maximum: 0.2 percent below");
    }
    // The options are checked.
    check(test::raises(
              [&] {
                  osc0.verify({R0, none, 5}, {safe({1, 0}, 1.3)},
                              alg::ReachAvoidKrylov{.krylov_step = 0});
              },
              Error::wrong_value),
          "krylov_step 0 is refused");
    return test::finish("linear_sys verify krylov");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
