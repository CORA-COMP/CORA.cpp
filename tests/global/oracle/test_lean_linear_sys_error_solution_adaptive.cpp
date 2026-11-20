// test_lean_linear_sys_error_solution_adaptive - the error solution of lin-adaptive against the
//
// CORALean oracle
//
// The radius of the error solution that lin-adaptive tunes by (priv_error_radius_float, the row
// sums of |G| from the last generator) is that of the oracle's errorSolutionAdaptiveFloat bit for
// bit at nearest, at a given order and without a static input. The public
// LinearSys::error_solution_adaptive cuts the series by the gain and takes the remainder from
// expm; it agrees with the oracle by tolerance (test_lean_linear_sys_steps_agreement).
//
// ORACLE: contDynamics/@linearSys/errorSolution_adaptive.m
// Authors:       Claude Opus 5.5
// Written:       20-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The sum of |G| of each coordinate from the last generator to the first (CORALean's sumUp).
Vec aux_radius(const Zonotope &Z) {
    const int64_t n = Z.dim(), h = Z.nr_generators();
    const Vec G = Z.generators().data();
    Vec radius(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        Vec row;
        for (int64_t l = 0; l < h; ++l) row.push_back(std::abs(G[l * n + i]));
        radius[i] = priv_sum_up(row);
    }
    return radius;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.errorSolution_adaptive", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 4})
            for (const double dt : {0.01, 0.05})
                for (const int eta : {2, 4, 6}) {
                    const Tensor A = rnd.matrix(n, n);
                    const Zonotope V = rnd.zonotope(n, n + 1);
                    const Zonotope theirs = lean::LinearSys(A)
                                                .error_solution_adaptive(V, std::nullopt, dt, eta)
                                                .gather()
                                                .first;
                    test::check(priv_error_radius_float(A, dt, eta, V) == aux_radius(theirs),
                                "errorSolution_adaptive: the radius, n = " + std::to_string(n) +
                                    ", eta = " + std::to_string(eta));
                }
    });
    return test::finish("lean linear_sys error_solution_adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
