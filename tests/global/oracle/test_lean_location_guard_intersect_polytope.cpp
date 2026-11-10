// test_lean_location_guard_intersect_polytope - the polytope guards of nondetGuard and conZonotope
//
// A halfspace guard and a hyperplane guard with a halfspace: the parallelotope (nondetGuard) or the
// reduced set (conZonotope) is cut by rows on the factors, the cut is boxed by k Fourier-Motzkin
// passes in each basis, and the boxes are joined and mapped back. The bundle is CORALean's
// `Location.guardIntersectNondetHFloat` and `Location.guardIntersectPolytopeHFloat`, bit for bit.
//
// ORACLE: hybridDynamics/@location/guardIntersect_nondetGuard.m
// ORACLE: hybridDynamics/@location/guardIntersect_conZonotope.m
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/oracle/location.h"
#include "hybrid_dynamics/location/private/priv.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A guard that cuts every set: rows a'x <= a'x0 + u through the common center region x0, and
/// with `hyperplane` also a'x = a'x0 for the first row.
Polytope aux_guard(test::oracle::Random &rnd, const Tensor &x0, int64_t n, bool hyperplane) {
    const Tensor A = rnd.matrix(2, n);
    std::vector<double> b(2);
    for (std::size_t i = 0; i < 2; ++i) {
        const Tensor row = A.slice(-2, int64_t(i), int64_t(i) + 1);
        b[i] = row.matmul(x0).item() + rnd.uniform(0.0, 0.3);
    }
    if (!hyperplane) return {A, Tensor::from_data(b, {2, 1})};
    const Tensor Ae = A.slice(-2, 0, 1);
    return {A, Tensor::from_data(b, {2, 1}), Ae, Ae.matmul(x0)};
}

/// Sets of n dimensions with centers near x0 and generators wide enough to cross the guard.
void aux_sets(test::oracle::Random &rnd, const Tensor &x0, int64_t n, std::vector<Zonotope> &sets,
              std::vector<Set> &reach) {
    for (int k = 0; k < 4; ++k) {
        sets.emplace_back(x0 + 0.1 * rnd.vector(n), rnd.matrix(n, n + 2));
        reach.emplace_back(sets.back());
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    // nondetGuard: the parallelotope of the sets, cut and boxed in two bases
    test::oracle::run("location.guardIntersect_nondetGuard", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3}) {
            for (const bool hyperplane : {false, true}) {
                const Tensor x0 = rnd.vector(n);
                std::vector<Zonotope> sets;
                std::vector<Set> reach;
                aux_sets(rnd, x0, n, sets, reach);
                const Polytope guard = aux_guard(rnd, x0, n, hyperplane);
                const std::vector<Tensor> bases = {Tensor::eye(n), rnd.matrix(n, n)};
                // no pass, one, and several (the oracle's cost grows like (factors)^k)
                for (const int k : {0, 2}) {
                    const std::optional<Set> mine =
                        priv_guard_intersect_nondet_guard(reach, guard, bases, k);
                    const lean::ZonoBundle theirs = lean::Location::guard_intersect_nondet_guard(
                        bases, sets.front(), sets, guard, k);
                    test::check(mine.has_value(), "nondetGuard: the cut is not empty");
                    if (mine) test::oracle::check(mine->as<ZonoBundle>(), theirs);
                }
            }
        }
    });
    // conZonotope: each reduced set cut and boxed, the boxes joined
    test::oracle::run("location.guardIntersect_conZonotope", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3}) {
            for (const bool hyperplane : {false, true}) {
                const Tensor x0 = rnd.vector(n);
                std::vector<Zonotope> sets;
                std::vector<Set> reach;
                aux_sets(rnd, x0, n, sets, reach);
                const Polytope guard = aux_guard(rnd, x0, n, hyperplane);
                const std::vector<Tensor> bases = {Tensor::eye(n), rnd.matrix(n, n)};
                for (const int k : {0, 2}) {
                    // reduced to order 1 first: n + 2 generators of n dimensions are order > 1
                    const std::optional<Set> mine =
                        priv_guard_intersect_con_zonotope(reach, guard, bases, 1, k);
                    const lean::ZonoBundle theirs = lean::Location::guard_intersect_con_zonotope(
                        bases, sets.front(), sets, guard, 1, k);
                    test::check(mine.has_value(), "conZonotope: the cut is not empty");
                    if (mine) test::oracle::check(mine->as<ZonoBundle>(), theirs);
                }
            }
        }
    });
    return test::finish("lean location guardIntersect with polytope guards");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
