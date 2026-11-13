// test_lean_nonlinear_sys_lin_reach_poly - one step of alg lin from a polynomial zonotope against
// the CORALean oracle, bit for bit
//
// The step of NonlinearSys::init_reach and post with a polynomial zonotope and alg lin at a linear
// step of CORALean's is CORALean's linReachPolyLoopFloat: the Hessian remainder over the zonotope
// of the time-interval set of a pass with the applied error in its input, accepted when the error
// is at most the applied one, the sets of one closed-form step from the polynomial zonotope, whose
// enclosure keeps the factors. Both sets, their exponents and identifiers and the error the next
// step starts from are the oracle's, bit for bit at nearest, in the order of the first occurrences
// of the merged columns (columnOrder first), on van der Pol with a cubic start set, an autonomous
// pendulum, TORA, ACC and a three-state system with products, over one to three steps in a row.
//
// ORACLE: contDynamics/@contDynamics/linReach.m
// ORACLE: contDynamics/@nonlinearSys/initReach.m
// ORACLE: contDynamics/@nonlinearSys/post.m
// ORACLE: contSet/@polyZonotope/enclose.m
// ORACLE: contSet/@polyZonotope/zonotope.m
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/nonlinear_sys.h"
#include "oracle_testing.h"

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vars = std::vector<Expr>;

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const Vars &x, const Vars &u) {
            return Vars{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

NonlinearSys pendulum() {
    return NonlinearSys(
        [](const Vars &x, const Vars &) { return Vars{x[1], -sin(x[0]) - 0.25 * x[1]}; }, 2, 0);
}

NonlinearSys tora() {
    return NonlinearSys(
        [](const Vars &x, const Vars &u) {
            return Vars{x[1], -x[0] + 0.0625 * sin(x[2]), x[3], u[0]};
        },
        4, 1);
}

NonlinearSys acc() {
    constexpr double mu = 0x1p-13;
    return NonlinearSys(
        [=](const Vars &x, const Vars &u) {
            return Vars{x[1], x[2], -2.0 * x[2] - 4.0 - mu * pow(x[1], 2),
                        x[4], x[5], -2.0 * x[5] + 2.0 * u[0] - mu * pow(x[4], 2)};
        },
        6, 1);
}

NonlinearSys lotka() {
    return NonlinearSys(
        [](const Vars &x, const Vars &) {
            return Vars{x[0] * (1 - x[1]), x[1] * (x[0] - 1), -x[2] + x[0] * x[1]};
        },
        3, 0);
}

/// The column of the values.
Tensor col(const std::vector<double> &v) { return Tensor::from_data(v, {int64_t(v.size()), 1}); }

/// The matrix of the rows, (rows, cols).
Tensor mat(int64_t rows, int64_t cols, const std::vector<double> &v) {
    return Tensor::from_data(v, {rows, cols});
}

alg::Linearization aux_alg(double dt, int eta, int rho) {
    alg::Linearization alg;
    alg.time_step = dt;
    alg.taylor_terms = eta;
    alg.zonotope_order = rho;
    alg.linear_step = alg::LinearStep::closed_form;
    alg.error_bound = alg::ErrorBound::loop;
    return alg;
}

/// Steps in a row from R: init_reach, then post (rho large, so that post reduces nothing), each
/// against the oracle's step from the same set and error.
void aux_chain(const std::string &name, const NonlinearSys &sys, const PolyZonotope &R,
               const std::optional<Zonotope> &U, double dt, int eta, int steps, int rho = 100000) {
    const alg::Linearization alg = aux_alg(dt, eta, rho);
    std::optional<lean::Zonotope> lean_U;
    if (U) lean_U = lean::Zonotope(*U);
    PolyNonlinearStep mine{R, R, Tensor()};
    PolyZonotope from = R;
    std::optional<Tensor> error;
    for (int k = 0; k < steps; ++k) {
        const std::string tag = name + " step " + std::to_string(k);
        const std::optional<lean::PolyLinReachResult> theirs =
            lean::NonlinearSys(sys).lin_reach_poly(lean_U, lean::PolyZonotope(from), alg, error,
                                                   k > 0);
        test::check(theirs.has_value(), tag + ": the oracle's step");
        if (!theirs) return;
        mine = k == 0 ? sys.init_reach(R, U, alg) : sys.post(mine, U, alg);
        oracle::check(tag + " time interval", mine.time_interval, theirs->time_interval);
        oracle::check(tag + " time point", mine.time_point, theirs->time_point);
        oracle::check(tag + " error", mine.error, theirs->abstrerr);
        from = mine.time_point;
        error = mine.error;
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("nonlinearSys.linReachPoly", [&](oracle::Random &) {
        const double s5 = 0x1p-5, s6 = 0x1p-6, s7 = 0x1p-7;
        // van der Pol, a cubic start set with a generator on two factors
        const PolyZonotope vdp(col({1.75, 1.25}), mat(2, 3, {s5, s6, 0, 0, s6, s5}),
                               mat(2, 1, {s7, 0}), {{1, 0}, {0, 1}, {2, 1}}, {1, 2});
        const Zonotope vdp_U(Tensor({0.0}), Tensor({{s6}}));
        aux_chain("van der Pol", van_der_pol(), vdp, vdp_U, s5, 4, 3);
        const PolyZonotope pend(col({1.0, 0.0}), mat(2, 2, {s5, 0, 0, s6}), Tensor::zeros({2, 0}),
                                {{1, 0}, {0, 1}}, {1, 2});
        aux_chain("pendulum", pendulum(), pend, std::nullopt, s5, 4, 2);
        const PolyZonotope tor(
            col({0.375, -0.375, -0.5, 0.5}),
            mat(4, 4, {s6, 0, 0, 0, 0, s6, 0, 0, 0, 0, s6, 0, 0, 0, 0, s6}), Tensor::zeros({4, 0}),
            {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}, {1, 2, 3, 4});
        aux_chain("TORA", tora(), tor, Zonotope(Tensor({0.0}), Tensor({{0.125}})), s5, 4, 2);
        const PolyZonotope lv(col({0.75, 1.0, 0.5}), mat(3, 3, {s6, 0, s7, 0, s6, 0, 0, s7, s6}),
                              mat(3, 1, {s7 / 2, 0, 0}), {{1, 0}, {0, 1}, {1, 1}}, {1, 2});
        aux_chain("Lotka", lotka(), lv, std::nullopt, s6, 5, 3);
        const PolyZonotope ac(
            col({90, 32, 0, 10, 30, 0}),
            mat(6, 3, {0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0.125, 0}),
            Tensor::zeros({6, 0}), {{1, 0}, {0, 1}, {1, 1}}, {1, 2});
        aux_chain("ACC", acc(), ac, Zonotope(Tensor({1.0}), Tensor({{0.25}})), 0x1p-4, 4, 2);
        // the reduction past rho * n generators: post boxes the shortest, as reduceCora does
        aux_chain("van der Pol rho 3", van_der_pol(), vdp, vdp_U, s5, 4, 4, 3);
        aux_chain("Lotka rho 2", lotka(), lv, std::nullopt, s6, 5, 3, 2);
        aux_chain("TORA rho 2", tora(), tor, Zonotope(Tensor({0.0}), Tensor({{0.125}})), s5, 4, 3,
                  2);
        aux_chain("ACC rho 2", acc(), ac, Zonotope(Tensor({1.0}), Tensor({{0.25}})), 0x1p-4, 4, 3,
                  2);
    });
    return test::finish("lean nonlinearSys lin reach poly");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
