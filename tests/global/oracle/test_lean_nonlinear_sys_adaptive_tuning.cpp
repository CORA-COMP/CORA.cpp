// test_lean_nonlinear_sys_adaptive_tuning - the tuning of lin-adaptive against the CORALean oracle
//
// The gain over a change of the step size, the gain that explains a measured error, the step size
// that minimizes the estimated size of the sets and the reduction of a zonotope by its diagonal
// share are arithmetic on numbers: they agree with the oracle at nearest bit for bit.
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA: contSet/@zonotope/reduce.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/oracle/nonlinear_sys.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The body at nearest on the default backend, for the oracle op `op` (skipped if it lacks it).
template <class F>
void aux_run(const std::string &op, F body) {
    if (!test::oracle::serves(op)) return;
    test::oracle::op_name() = op;
    const lean::DTypeScope dtype("nearest");
    test::oracle::Random rnd(11);
    body(rnd);
}

/// Exactly equal, as the doubles of the two sides.
void aux_same(const std::string &what, double mine, double theirs) {
    test::check(test::oracle::agree(what, mine, theirs), what + ": the oracle's value");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    const alg::LinearizationAdaptive alg;

    // the gain from a step size to a smaller one: the step sizes are a ratio of decr^k apart
    aux_run("nonlinearSys.varphiTotalAdaptive", [&](test::oracle::Random &rnd) {
        for (int k = 0; k < 20; ++k) {
            const double curr = rnd.uniform(0.01, 1.0);
            const double prev = curr * rnd.uniform(0.2, 5.0);
            const double lim = rnd.uniform(0.0, 0.5), varphi = rnd.uniform(0.5, 0.95);
            aux_same("varphiTotal", priv_varphi_total(varphi, lim, prev, curr, alg.decr_factor),
                     lean::NonlinearSys::varphi_total_adaptive(alg, varphi, lim, prev, curr));
        }
    });

    // the gain whose total scaling matches the errors of two step sizes
    aux_run("nonlinearSys.varphiEstAdaptive", [&](test::oracle::Random &rnd) {
        for (int k = 0; k < 10; ++k) {
            const Tensor error_h = rnd.matrix(3, 1).abs() + 0.1,
                         error = error_h * rnd.uniform(0.2, 1.0);
            const std::vector<bool> mask{true, k % 3 != 0, true};
            const std::vector<char> flags(mask.begin(), mask.end());
            const double deltat = rnd.uniform(0.01, 0.1), horizon = deltat * rnd.uniform(1.5, 20);
            const int minorder = k % 3;
            aux_same("varphiEst",
                     priv_varphi_est(horizon, deltat, error_h.data(), error.data(), alg.decr_factor,
                                     flags, minorder, alg.fuel),
                     lean::NonlinearSys::varphi_est_adaptive(alg, error_h, error, horizon, deltat,
                                                             mask, minorder));
        }
    });

    // the step size of least estimated size
    aux_run("nonlinearSys.optimalDeltatAdaptive", [&](test::oracle::Random &rnd) {
        for (int k = 0; k < 10; ++k) {
            const double r_radius = rnd.uniform(0.1, 3.0), rerr1 = rnd.uniform(1e-4, 1e-1);
            const double deltat = rnd.uniform(0.01, 0.5), varphimin = rnd.uniform(0.0, 0.5);
            const double zeta_p = rnd.uniform(0.5, 2.0), mu = alg.decr_factor;
            aux_same(
                "optimalDeltat",
                priv_optimal_delta_t(mu, alg.zeta_z, r_radius, rerr1, deltat, varphimin, zeta_p),
                lean::NonlinearSys::optimal_delta_t_adaptive(alg, alg.zeta_z, r_radius, rerr1,
                                                             deltat, varphimin, zeta_p));
        }
    });

    // the reduction by the diagonal share keeps what it must and replaces the rest by a box
    aux_run("zonotope.reduce", [&](test::oracle::Random &rnd) {
        for (int k = 0; k < 10; ++k) {
            // three generators of each size, so that some are boxed and some kept
            std::vector<double> g = rnd.matrix(3, 9).data();
            for (std::size_t i = 0; i < g.size(); ++i)
                g[i] *= i % 9 < 3 ? 1.0 : (i % 9 < 6 ? 0.01 : 0.001);
            const Tensor G = Tensor::from_data(g, {3, 9});
            const Zonotope Z(rnd.vector(3), G);
            const double share = k % 2 == 0 ? 0.05 : 0.5;
            test::check(test::oracle::agree("reduce", Z.reduce_adaptive(share),
                                            lean::Zonotope(Z).reduce_adaptive(share)),
                        "zonotope.reduce adaptive: the oracle's zonotope");
        }
    });

    return test::finish("lean nonlinearSys adaptive tuning");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
