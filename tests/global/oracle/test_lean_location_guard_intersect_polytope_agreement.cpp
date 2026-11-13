// test_lean_location_guard_intersect_polytope_agreement - certificate boxes of polytope guards
//
// CORA.cpp's simplex proposes the dual solution of each program in floating point, the oracle's
// exact simplex at rationals proposes its own; both are checked by weak duality in directed order,
// so the boxes agree to the last bits of the multipliers, not bit for bit. The hulls of the
// bundle's members agree to a relative 1e-8 on random sets (the bit-exact case, where the dual
// solution is unique and dyadic, is in `test_lean_location_guard_intersect_polytope`).
//
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

/// The hulls of the members of a bundle agree to a relative 1e-8 (the oracle's multipliers are the
/// exact simplex's, CORA.cpp's the float simplex's).
void aux_check_hulls(const std::optional<Set> &mine, const lean::ZonoBundle &theirs) {
    test::check(mine.has_value(), "the cut is not empty");
    if (!mine) return;
    const ZonoBundle &bundle = mine->as<ZonoBundle>();
    test::check(bundle.zonotopes().size() == theirs.zonotopes().size(), "one box per basis");
    for (std::size_t i = 0; i < bundle.zonotopes().size(); ++i)
        test::oracle::check_hull(bundle.zonotopes()[i], theirs.zonotopes()[i], 1e-8);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    // random sets: the hulls agree to the last bits of the multipliers
    test::oracle::run("location.guardIntersect_nondetGuard", [](test::oracle::Random &rnd) {
        // random sets cut by a hyperplane with a halfspace, in two bases
        const int64_t n = 3;
        const Tensor x0 = rnd.vector(n);
        std::vector<Zonotope> sets;
        std::vector<Set> reach;
        aux_sets(rnd, x0, n, sets, reach);
        const Polytope guard = aux_guard(rnd, x0, n, true);
        const std::vector<Tensor> bases = {Tensor::eye(n), rnd.matrix(n, n)};
        aux_check_hulls(
            priv_guard_intersect_nondet_guard(reach, guard, bases, alg::CutBox::cert, 0, true),
            lean::Location::guard_intersect_nondet_guard(bases, sets.front(), sets, guard, 0,
                                                         "cert"));
    });
    test::oracle::run("location.guardIntersect_conZonotope", [](test::oracle::Random &rnd) {
        // the same sets, each cut and boxed on its own
        const int64_t n = 3;
        const Tensor x0 = rnd.vector(n);
        std::vector<Zonotope> sets;
        std::vector<Set> reach;
        aux_sets(rnd, x0, n, sets, reach);
        const Polytope guard = aux_guard(rnd, x0, n, true);
        const std::vector<Tensor> bases = {Tensor::eye(n), rnd.matrix(n, n)};
        aux_check_hulls(
            priv_guard_intersect_con_zonotope(reach, guard, bases, 1, alg::CutBox::cert, 0, true),
            lean::Location::guard_intersect_con_zonotope(bases, sets.front(), sets, guard, 1, 0,
                                                         "cert"));
    });
    return test::finish("lean location guardIntersect certificate boxes");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
