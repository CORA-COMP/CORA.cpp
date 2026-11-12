// test_lean_nonlinear_sys_poly_adaptive - poly-adaptive against CORALean's reachPolyAdaptiveFloat
//
// The ops nonlinearSys.linReachPolyAdaptivePass and nonlinearSys.reachPolyAdaptive at nearest, on
// the van der Pol oscillator with an input (the system of the oracle's fixture 3201) and a small
// quadratic system whose third-order tensor is empty. The values that tuning reads off a pass (the
// point of linearization xs, the matrix A, the zero generators of the error) are the oracle's bit
// for bit, as are the first step sizes of a run, the true error and the radius of the solutions of
// the errors (the abstraction error is CORALean's operation for operation, priv_poly_quad.cpp). The
// boxes of the window and of the end set (PolyZonotope.interval, error box included) agree to 1e-5
// only: CORALean keeps the center shift of zonotopeEven in the error box of VerrorDyn, where it is
// part of the varying input, and CORA.cpp's zonotopes have no error box (docs/oracle-requests.md).
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"
#include "cora/global/oracle/nonlinear_sys.h"
#include "oracle_testing.h"

#include <iostream>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

NonlinearSys quadratic() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{-x[0] + 0.5 * x[1] * x[2], -x[1] + 0.3 * x[0] * x[0] - x[2],
                                     x[1] - 0.8 * x[2] - 0.2 * x[0] * x[1]};
        },
        3, 1);
}

/// A system with its start set, input set, step size and whether its third derivatives vanish.
struct Case {
    const char *name;
    NonlinearSys sys;
    Zonotope R, U;
    double dt;
    bool empty;
};

std::vector<Case> cases() {
    return {{"van der Pol", van_der_pol(),
             Zonotope(Tensor({1.75, 1.25}), Tensor({{0.03125, 0.0}, {0.0, 0.03125}})),
             Zonotope(Tensor({0.0}), Tensor({{0.015625}})), 1.0 / 1024, false},
            {"quadratic", quadratic(),
             Zonotope(Tensor({1.0, 0.5, 0.2}),
                      Tensor({{0.05, 0.0, 0.0}, {0.0, 0.05, 0.0}, {0.0, 0.0, 0.05}})),
             Zonotope(Tensor({0.0}), Tensor::zeros({1, 0})), 0.002, true}};
}

/// PolyZonotope.interval of CORALean: the center less and plus the row sums of the dependent and
/// of the independent generators in turn, each sum nested to the right.
Interval aux_free_box(const PolyZonotope &Z) {
    const std::vector<double> c = Z.center().data(), G = Z.generators().data(),
                              GI = Z.independent_generators().data();
    const std::size_t n = c.size(), h = std::size_t(Z.num_dependent()),
                      q = std::size_t(Z.independent_generators().shape()[1]);
    std::vector<double> lo(n), hi(n);
    for (std::size_t i = 0; i < n; ++i) {
        double rg = 0, rgi = 0;
        for (std::size_t l = h; l-- > 0;) rg = std::abs(G[i * h + l]) + rg;
        for (std::size_t l = q; l-- > 0;) rgi = std::abs(GI[i * q + l]) + rgi;
        lo[i] = (c[i] - rg) - rgi;
        hi[i] = (c[i] + rg) + rgi;
    }
    const auto N = static_cast<int64_t>(n);
    return {Tensor::like(Z.center(), lo, {N, 1}), Tensor::like(Z.center(), hi, {N, 1})};
}

/// The boxes agree to rel (relative to 1 and the larger width).
bool aux_close_box(const Interval &mine, const Interval &theirs, double rel) {
    const std::vector<double> a = mine.inf().data(), b = theirs.inf().data(), c = mine.sup().data(),
                              d = theirs.sup().data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double scale = rel * (1 + std::max(c[i] - a[i], d[i] - b[i]));
        ok = std::abs(a[i] - b[i]) <= scale && std::abs(c[i] - d[i]) <= scale;
    }
    return ok;
}

/// The values agree to rel (relative to the larger of the two and 1).
bool aux_close(const std::vector<double> &a, const std::vector<double> &b, double rel) {
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i)
        ok = std::abs(a[i] - b[i]) <= rel * (1 + std::max(std::abs(a[i]), std::abs(b[i])));
    return ok;
}

/// The body at nearest on the default backend, for the oracle op `op` (skipped if it lacks it).
template <class F>
void aux_run(const std::string &op, F body) {
    if (!test::oracle::serves(op)) return;
    test::oracle::op_name() = op;
    const lean::DTypeScope dtype("nearest");
    body();
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    // the fixture's Taylor order and fuel: the oracle keeps every generator and is slow
    alg::PolynomializationAdaptive alg;
    alg.taylor_terms = 6;
    alg.fuel = 20;

    // one pass: the values the tuning reads exactly, the sets and the errors to the tolerances
    aux_run("nonlinearSys.linReachPolyAdaptivePass", [&] {
        // both systems from the same start set, input set and step size
        for (Case &c : cases()) {
            const lean::NonlinearSys oracle(c.sys);
            const std::optional<Zonotope> U = c.U;
            const std::vector<double> adm(std::size_t(c.sys.dim()), 0.0);
            const std::string b = c.name;
            const auto mine =
                priv_lin_reach_poly_adaptive_pass(c.sys, U, alg, c.dt, PolyZonotope(c.R), adm);
            const auto theirs = oracle.lin_reach_poly_adaptive_pass(
                U, c.R, c.dt, priv_column(c.R.center(), adm), alg, c.empty);
            test::check(mine.has_value() && theirs.has_value(), b + ": both converge");
            if (!mine || !theirs) continue;
            test::check(test::oracle::agree(b + " xs", priv_column(c.R.center(), mine->xs),
                                            lean::Tensor(theirs->xs)),
                        b + ": xs agrees with the oracle's");
            test::check(test::oracle::agree(b + " A", mine->A, lean::Tensor(theirs->A)),
                        b + ": A agrees with the oracle's");
            // the zero generators, then the boxes of the sets and the errors
            std::vector<bool> gzero(mine->gzero.begin(), mine->gzero.end());
            test::check(gzero == theirs->g_zero, b + ": the zero generators of the error");
            const Interval hull_i = aux_free_box(mine->Rti), hull_p = aux_free_box(mine->Rtp);
            test::check(aux_close(mine->true_error, theirs->true_error.data(), 0.0),
                        b + ": the true error");
            test::check(aux_close(mine->abstrerr, theirs->abstrerr.data(), 0.0),
                        b + ": the radius of the solutions of the errors");
            test::check(aux_close_box(hull_i, theirs->time_interval, 1e-5),
                        b + ": the window over the step agrees with the oracle's");
            test::check(aux_close_box(hull_p, theirs->time_point, 1e-5),
                        b + ": the set at the end of the step agrees with the oracle's");
        }
    });

    // a run of one step (the oracle takes minutes per step): its size and the set at its end
    aux_run("nonlinearSys.reachPolyAdaptive", [&] {
        Case c = std::move(cases().front());
        const lean::NonlinearSys oracle(c.sys);
        const std::string b = c.name;
        const alg::PolynomializationAdaptive &one = alg;
        const PolyReachSet mine = c.sys.reach(
            {.initial_set = PolyZonotope(c.R), .input_set = c.U, .t_final = 0.001}, one);
        const lean::PolyAdaptiveRun theirs =
            oracle.reach_poly_adaptive(c.R, c.U, 0.001, 1, one, c.empty);
        std::cerr << b << " steps " << mine.time.size() - 1 << " vs " << theirs.steps.size() << ": "
                  << mine.time[1] << " vs " << theirs.steps.front() << "\n";
        test::check(!theirs.steps.empty() &&
                        aux_close({mine.time[1]}, {theirs.steps.front()}, 5e-2),
                    b + ": the size of the first step");
    });
    return test::finish("lean nonlinear_sys poly adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
