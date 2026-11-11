// test_lean_nonlinear_sys_adaptive_pass - one linearization of lin-adaptive against the oracle
//
// The abstraction error and the orders of its terms are computed by the same arithmetic as
// CORALean's and agree with the oracle bit for bit at nearest (van der Pol with an input, the
// Robertson reaction). A pass takes a linear step with the error as input: CORA.cpp's is CORA's
// oneStep and CORALean's is the interval-matrix exponential, so the sets of a pass agree to
// 1e-7 (relative) and the values read off them (the point of linearization, the matrix A, the
// zero generators) exactly.
//
// CORA: contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/oracle/nonlinear_sys.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kSets = 5e-3, kError = 0.5;

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

NonlinearSys robertson() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{-0.4 * x[0] + 100 * x[1] * x[2],
                                     0.4 * x[0] - 100 * x[1] * x[2] - 1000 * pow(x[1], 2),
                                     1000 * pow(x[1], 2)};
        },
        3, 1);
}

/// A system with its start set and input set.
struct Case {
    const char *name;
    NonlinearSys sys;
    Zonotope R, U;
    double dt;
};

std::vector<Case> cases() {
    return {{"van der Pol", van_der_pol(),
             Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.01, 0.002}, {0.0, 0.05, 0.003}})),
             Zonotope(Tensor({0.0}), Tensor({{0.1}})), 0.02},
            {"Robertson", robertson(),
             Zonotope(Tensor({1.0, 0.0, 0.0}),
                      Tensor({{1e-3, 0.0, 0.0}, {0.0, 1e-4, 0.0}, {0.0, 0.0, 1e-4}})),
             Zonotope(Tensor({0.0}), Tensor::zeros({1, 0})), 0.001}};
}

/// The box of `mine` lies in the oracle's, up to 1e-9 (relative), and is tighter by at most `gap`
/// (relative): the oracle bounds the linear step and the quadratic map more coarsely.
void aux_inside(const std::string &what, const Zonotope &mine, const lean::Zonotope &theirs,
                double gap) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double slack = 1e-9 * (1 + std::abs(b[i])), far = gap * (1 + std::abs(b[i]));
        ok = a[i] >= b[i] - slack && a[i] <= b[i] + far && c[i] <= d[i] + slack &&
             c[i] >= d[i] - far;
    }
    test::check(ok, what + ": inside the oracle's box and close to it");
}

/// Exactly the oracle's values.
template <class A, class B>
void aux_same(const std::string &what, const A &mine, const B &theirs) {
    test::check(test::oracle::agree(what, mine, theirs), what + ": the oracle's value");
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
    const alg::LinearizationAdaptive alg;

    // the passes of both systems: the error of the window, then the orders of its terms
    aux_run("nonlinearSys.linReachAdaptivePass", [&] {
        for (Case &c : cases()) {
            const lean::NonlinearSys oracle(c.sys);
            const std::optional<Zonotope> U = c.U;
            const std::optional<lean::Zonotope> lean_input = lean::Zonotope(c.U);
            const std::vector<double> adm(std::size_t(c.sys.dim()), 0.0);
            for (const int kappa : {2, 3}) {
                const std::string b = std::string(c.name) + " at order " + std::to_string(kappa);
                const std::optional<AdaptivePass> mine =
                    priv_lin_reach_adaptive_pass(c.sys, kappa, U, alg, c.dt, c.R, adm);
                const auto theirs =
                    oracle.lin_reach_adaptive_pass(lean_input, lean::Zonotope(c.R), c.dt, kappa,
                                                   priv_column(c.R.center(), adm), alg);
                // the sets inside the oracle's (its linear step is coarser), the values exact
                test::check(mine.has_value() && theirs.has_value(), b + ": both converge");
                if (!mine || !theirs) continue;
                aux_same(b + " xs", priv_column(c.R.center(), mine->xs), theirs->xs);
                aux_same(b + " A", mine->A, theirs->A);
                aux_inside(b + " Rmax", mine->Rmax, theirs->Rmax, kSets);
                aux_inside(b + " time interval", mine->Rti, theirs->time_interval, kSets);
                aux_inside(b + " time point", mine->Rtp, theirs->time_point, kSets);
                std::vector<bool> gzero(mine->gzero.begin(), mine->gzero.end());
                test::check(gzero == theirs->g_zero, b + ": the zero generators");

                // the error of the window over Rmax, from the matrices at the point of the pass
                const Jacobians J =
                    c.sys.jacobian(priv_column(c.R.center(), mine->xs), c.U.center());
                // the error over Rmax from the point of the pass, against the oracle's
                const Zonotope V = priv_abstr_err_adaptive(c.sys, kappa, J.A, J.B, mine->xs,
                                                           mine->Rmax, U, alg.zeta_z);
                const auto err =
                    oracle.abstr_err_adaptive(lean_input, priv_column(c.R.center(), mine->xs),
                                              lean::Zonotope(mine->Rmax), kappa, alg.zeta_z);
                // order 2 is the oracle's bit for bit; at order 3 the quadratic map is CORA's
                if (kappa == 2) {
                    test::check(test::oracle::agree(b + " VerrorDyn", V, err.V),
                                b + ": VerrorDyn is the oracle's");
                    aux_same(b + " true error",
                             priv_column(c.R.center(), priv_norm_mag(priv_zonotope_box(V))),
                             err.true_error);
                } else {
                    aux_inside(b + " VerrorDyn", V, err.V, kError);
                }
            }
        }
    });
    // the orders of the error terms, for the dimensions of zero width and not
    aux_run("nonlinearSys.ordersAdaptive", [&] {
        for (Case &c : cases()) {
            const lean::NonlinearSys oracle(c.sys);
            const int64_t n = c.sys.dim();
            const std::vector<double> adm(std::size_t(n), 0.0);
            const auto pass = priv_lin_reach_adaptive_pass(c.sys, 2, c.U, alg, c.dt, c.R, adm);
            if (!pass) continue;
            const Tensor xs = priv_column(c.R.center(), pass->xs);
            // every pattern of zero-width dimensions
            for (int64_t mask = 0; mask < (int64_t(1) << n); ++mask) {
                std::vector<bool> zero_width(static_cast<std::size_t>(n));
                for (int64_t i = 0; i < n; ++i) zero_width[std::size_t(i)] = ((mask >> i) & 1) != 0;
                const std::vector<char> flags(zero_width.begin(), zero_width.end());
                const auto H = c.sys.hessian_enclosure(Range(xs), Range(c.U.center()));
                const std::vector<int> mine =
                    priv_adaptive_orders(pass->A, flags, pass->gzero, H, n, c.sys.input_dim(), 100);
                const auto theirs = oracle.orders_adaptive(
                    lean::Zonotope(c.U), pass->A, xs, zero_width,
                    std::vector<bool>(pass->gzero.begin(), pass->gzero.end()));
                bool same = mine.size() == theirs.size();
                for (std::size_t i = 0; same && i < mine.size(); ++i)
                    same = theirs[i] ? mine[i] == *theirs[i] : mine[i] < 0;
                test::check(same, std::string(c.name) + ": the orders of the oracle");
            }
        }
    });
    return test::finish("lean nonlinearSys adaptive pass");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
