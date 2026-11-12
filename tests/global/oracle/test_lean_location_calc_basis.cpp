// test_lean_location_calc_basis - the box and pca bases against the CORALean oracle
//
// The `box` basis is the identity of the state dimension, one per entry of `enclose`. The `pca`
// basis is CORALean's eigenvectors of the generators' covariance (ten Jacobi sweeps, the generators
// projected on a hyperplane guard), bit for bit; the oracle has no float definition of `flow`,
// which stays CORA's.
//
// ORACLE: hybridDynamics/@location/calcBasis.m
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/location.h"
#include "hybrid_dynamics/location/private/priv.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("location.calcBasis", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 5}) {
            const Location loc("l", Polytope::inf(n), {}, LinearSys(rnd.matrix(n, n)));
            const Polytope guard = Polytope::inf(n);
            const std::vector<Set> R = {rnd.zonotope(n, n)};
            const std::vector<Tensor> mine =
                priv_calc_basis(loc, R, guard, {alg::Enclose::box, alg::Enclose::box},
                                Zonotope(loc.c(), loc.c().zeros_like({n, 0})));
            const std::vector<Tensor> theirs = lean::Location::calc_basis(n, {"box", "box"});
            for (std::size_t k = 0; k < mine.size(); ++k) test::oracle::check(mine[k], theirs[k]);
        }
        // pca: the sets of a crossing, with and without a hyperplane guard
        for (const int64_t n : {2, 3, 5}) {
            const Location loc("l", Polytope::inf(n), {}, LinearSys(rnd.matrix(n, n)));
            const Zonotope U(loc.c(), loc.c().zeros_like({n, 0}));
            std::vector<Zonotope> sets;
            std::vector<Set> R;
            for (int k = 0; k < 3; ++k) {
                sets.push_back(rnd.zonotope(n, n + 2));
                R.emplace_back(sets.back());
            }
            // free guard
            const std::vector<Tensor> mine = priv_calc_basis(
                loc, R, Polytope::inf(n), {alg::Enclose::box, alg::Enclose::pca}, U);
            const std::vector<Tensor> theirs =
                lean::Location::calc_basis(n, {"box", "pca"}, sets, std::nullopt);
            for (std::size_t k = 0; k < mine.size(); ++k) test::oracle::check(mine[k], theirs[k]);
            // a hyperplane guard
            Tensor normal = rnd.vector(n);
            const Polytope guard(Tensor::zeros({0, n}), Tensor::zeros({0, 1}),
                                 normal.matrix_transpose(), normal.full_like({1, 1}, 0.1));
            const std::vector<Tensor> mine_h =
                priv_calc_basis(loc, R, guard, {alg::Enclose::pca}, U);
            const std::vector<Tensor> theirs_h =
                lean::Location::calc_basis(n, {"pca"}, sets, normal);
            test::oracle::check(mine_h[0], theirs_h[0]);
        }
    });
    return test::finish("lean location calcBasis");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
