// test_lean_location_guard_intersect_zono_girard - zonoGirard against the CORALean oracle
//
// The box of the cut by a hyperplane.
// For each basis the cuts of the sets are joined as the box [-U(-l), U(l)] of the dual bounds, and
// the box is mapped back by the basis, plus the box of (I - B B') x where B B' is not exactly the
// identity (a random basis, the pca basis): CORALean's `Location.guardIntersectZonoGirardFloat`.
// The bounds of the guard's inequalities are applied afterwards by CORA.cpp and are not compared.
//
// ORACLE: hybridDynamics/@location/guardIntersect_zonoGirard.m
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/oracle/location.h"
#include "hybrid_dynamics/location/private/priv.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("location.guardIntersect_zonoGirard", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3}) {
            for (int trial = 0; trial < 4; ++trial) {
                // sets centered near the hyperplane x'normal = gamma so that it cuts them
                Tensor normal = rnd.vector(n);
                normal = normal / normal.square().sum().sqrt().item();
                const double gamma = rnd.uniform(-0.2, 0.2);
                std::vector<Zonotope> sets;
                std::vector<Set> reach;
                for (int k = 0; k < 4; ++k) {
                    sets.push_back(rnd.zonotope(n, n + 2));
                    reach.emplace_back(sets.back());
                }
                // two bases: the identity and a random one
                const Polytope guard(Tensor::zeros({0, n}), Tensor::zeros({0, 1}),
                                     normal.matrix_transpose(), normal.full_like({1, 1}, gamma));
                // and the pca basis of the sets, orthogonal up to rounding
                const Location loc("l", Polytope::inf(n), {}, LinearSys(rnd.matrix(n, n)));
                const Tensor pca =
                    priv_calc_basis(loc, reach, guard, {alg::Enclose::pca},
                                    Zonotope(loc.c(), loc.c().zeros_like({n, 0})))[0];
                const std::vector<Tensor> bases = {Tensor::eye(n), rnd.matrix(n, n), pca};
                const std::optional<Set> mine =
                    priv_guard_intersect_zono_girard(reach, guard, bases, true);
                const lean::ZonoBundle theirs = lean::Location::guard_intersect_zono_girard(
                    bases, sets.front(), sets, normal, gamma);
                // an empty cut is the dropped part, which CORALean does not define
                if (!mine) continue;
                test::oracle::check(mine->as<ZonoBundle>(), theirs);
            }
        }
    });
    return test::finish("lean location guardIntersect_zonoGirard");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
