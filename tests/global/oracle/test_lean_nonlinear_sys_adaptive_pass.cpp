// test_lean_nonlinear_sys_adaptive_pass - one linearization of lin-adaptive against the oracle
//
// A pass at a fixed step size is CORALean's linReachAdaptivePassFloat bit for bit at nearest on
// the lean backend, at the tensor orders 2 and 3 (van der Pol with an input, the Robertson
// reaction), with the linear step of CORALean's reachFloatInput (coralean) and of the closed form
// (the default): the point of linearization, the matrix A, the sets (Rmax, the window, the end
// set), the error the loop accepts, the radius of its solution that the tuning reads
// (priv_error_radius_float, errorSolutionAdaptiveFloat) and the zero generators. The error over
// Rmax is the oracle's at both orders: at order 3 the quadratic map is CORA's of a zonotope
// (quadMapCora). The orders of its terms agree for every pattern of zero-width dimensions.
//
// Short horizon: the closed-form step runs on van der Pol only; the oracle needs 25 minutes for it
// on Robertson's three states (the code does not depend on the dimension).
//
// ORACLE: contDynamics/@nonlinearSys/linReach_adaptive.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       10-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/oracle/nonlinear_sys.h"
#include "oracle_testing.h"

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
             Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.01, 0.05}, {0.002, 0.003}})),
             Zonotope(Tensor({0.0}), Tensor({{0.1}})), 0.02},
            {"Robertson", robertson(),
             Zonotope(Tensor({1.0, 0.0, 0.0}),
                      Tensor({{1e-3, 0.0, 0.0}, {0.0, 1e-4, 0.0}, {0.0, 0.0, 1e-4}})),
             Zonotope(Tensor({0.0}), Tensor::zeros({0, 1})), 0.001}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    for (const alg::LinearStep step : {alg::LinearStep::coralean, alg::LinearStep::closed_form}) {
        alg::LinearizationAdaptive alg;
        alg.linear_step = step;
        const bool closed = step == alg::LinearStep::closed_form;

        // the passes of both systems: the sets, then the error of the window over Rmax
        test::oracle::run("nonlinearSys.linReachAdaptivePass", [&](test::oracle::Random &) {
            for (Case &c : cases()) {
                // the oracle's closed form takes 25 minutes on Robertson (see the header)
                if (closed && c.sys.dim() > 2) continue;
                const lean::NonlinearSys oracle(c.sys);
                const std::optional<Zonotope> U = c.U;
                const std::optional<lean::Zonotope> lean_input = lean::Zonotope(c.U);
                const std::vector<double> adm(std::size_t(c.sys.dim()), 0.0);
                for (const int kappa : {2, 3}) {
                    const std::string b = std::string(c.name) +
                                          (closed ? " closed form" : " coralean") + " at order " +
                                          std::to_string(kappa);
                    const std::optional<AdaptivePass> mine =
                        priv_lin_reach_adaptive_pass(c.sys, kappa, U, alg, c.dt, c.R, adm);
                    const auto theirs = oracle.lin_reach_adaptive_pass(
                        lean_input, lean::Zonotope(c.R), c.dt, kappa,
                        // layout: #163 step 2
                        priv_column(detail::cora_center(c.R), adm), alg);
                    test::check(mine.has_value() && theirs.has_value(), b + ": both converge");
                    if (!mine || !theirs) continue;
                    // the values and the sets of the pass
                    const Tensor &like = detail::cora_center(c.R);
                    test::oracle::check(b + " xs", priv_column(like, mine->xs), theirs->xs);
                    test::oracle::check(b + " A", mine->A, theirs->A);
                    test::oracle::check(b + " Rmax", mine->Rmax, theirs->Rmax);
                    test::oracle::check(b + " time interval", mine->Rti, theirs->time_interval);
                    test::oracle::check(b + " time point", mine->Rtp, theirs->time_point);
                    test::oracle::check(b + " true error", priv_column(like, mine->true_error),
                                        theirs->true_error);
                    test::oracle::check(b + " abstrerr", priv_column(like, mine->abstrerr),
                                        theirs->abstrerr);
                    std::vector<bool> gzero(mine->gzero.begin(), mine->gzero.end());
                    test::check(gzero == theirs->g_zero, b + ": the zero generators");

                    // the error over Rmax from the point of the pass, against the oracle's
                    const Jacobians J =
                        c.sys.jacobian(priv_column(like, mine->xs), detail::cora_center(c.U));
                    const Zonotope V = priv_abstr_err_adaptive(c.sys, kappa, J.A, J.B, mine->xs,
                                                               mine->Rmax, U, alg.zeta_z);
                    const auto err =
                        oracle.abstr_err_adaptive(lean_input, priv_column(like, mine->xs),
                                                  lean::Zonotope(mine->Rmax), kappa, alg.zeta_z);
                    test::oracle::check(b + " VerrorDyn", V, err.V);
                    test::oracle::check(b + " true error of V",
                                        priv_column(like, priv_norm_mag(priv_zonotope_box(V))),
                                        err.true_error);
                }
            }
        });
        // the orders of the error terms, for the dimensions of zero width and not
        test::oracle::run("nonlinearSys.ordersAdaptive", [&](test::oracle::Random &) {
            for (Case &c : cases()) {
                const lean::NonlinearSys oracle(c.sys);
                const int64_t n = c.sys.dim();
                const std::vector<double> adm(std::size_t(n), 0.0);
                const auto pass = priv_lin_reach_adaptive_pass(c.sys, 2, c.U, alg, c.dt, c.R, adm);
                if (!pass) continue;
                // layout: #163 step 2
                const Tensor xs = priv_column(detail::cora_center(c.R), pass->xs);
                // every pattern of zero-width dimensions
                for (int64_t mask = 0; mask < (int64_t(1) << n); ++mask) {
                    std::vector<bool> zero_width(static_cast<std::size_t>(n));
                    for (int64_t i = 0; i < n; ++i)
                        zero_width[std::size_t(i)] = ((mask >> i) & 1) != 0;
                    const std::vector<char> flags(zero_width.begin(), zero_width.end());
                    // the orders from the Hessian enclosure at the point of the pass
                    const auto H =
                        c.sys.hessian_enclosure(Range(xs), Range(detail::cora_center(c.U)));
                    const std::vector<int> mine = priv_adaptive_orders(
                        pass->A, flags, pass->gzero, H, n, c.sys.input_dim(), 100);
                    // the oracle's orders for the same pattern
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
    }
    return test::finish("lean nonlinearSys adaptive pass");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
