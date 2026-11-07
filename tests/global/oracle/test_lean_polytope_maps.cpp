// test_lean_polytope_maps - the maps and predicates of Polytope that the CORALean oracle serves
//
// Bit for bit where CORALean builds the same set: origin, minus, lift and projectHighDim of
// halfspace polytopes; plus, mtimes, or, project and projectHighDim of vertex polytopes (CORALean
// lists every vertex, CORA.cpp the extreme ones: compared after compact 'V'); the Booleans
// isBounded, isFullDim, isequal and representsa; the bound of supportFunc; as a one-sided proof,
// isIntersecting (a proof of disjointness). Not compared: zonotope (CORALean's enclosure is another
// zonotope than the one around the covariance ellipsoid) and zonoBundle (one member in CORALean,
// one per facet pair here).
//
// ORACLE: contSet/@polytope/origin.m
// ORACLE: contSet/@polytope/isBounded.m
// ORACLE: contSet/@polytope/isFullDim.m
// ORACLE: contSet/@polytope/isequal.m
// ORACLE: contSet/@polytope/representsa_.m
// ORACLE: contSet/@polytope/minus.m
// ORACLE: contSet/@polytope/lift_.m
// ORACLE: contSet/@polytope/projectHighDim.m
// ORACLE: contSet/@polytope/mtimes.m
// ORACLE: contSet/@polytope/plus.m
// ORACLE: contSet/@polytope/or.m
// ORACLE: contSet/@polytope/project.m
// ORACLE: contSet/@polytope/supportFunc_.m
// ORACLE: contSet/@polytope/isIntersecting_.m
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/polytope.h"
#include "cora/global/tensor/tolerance.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// k points of n dimensions on a sphere about a center: all extreme.
Tensor aux_sphere(oracle::Random &rnd, int64_t n, int64_t k, double shift = 0.0) {
    std::vector<double> x = rnd.matrix(n, k).data();
    for (int64_t c = 0; c < k; ++c) {
        double norm = 0.0;
        for (int64_t i = 0; i < n; ++i) norm += x[i * k + c] * x[i * k + c];
        for (int64_t i = 0; i < n; ++i) x[i * k + c] = shift + x[i * k + c] / std::sqrt(norm);
    }
    return Tensor::from_data(x, {n, k});
}

/// The box |x| <= 2 and m random rows with offset 3: nonempty and bounded.
Polytope aux_halfspaces(oracle::Random &rnd, int64_t n, int64_t m) {
    // the negative identity with +0 off the diagonal, as the oracle prints it
    const Tensor eye = Tensor::eye(n), negative = eye * 0.0 - eye;
    return {Tensor::concat({eye, negative, rnd.matrix(m, n)}, 0),
            Tensor::concat({Tensor::ones({2 * n, 1}) * 2.0, Tensor::ones({m, 1}) * 3.0}, 0)};
}

/// The vertices of a result, reduced to the extreme ones as CORA.cpp keeps them.
Polytope aux_extreme(const Polytope &P) { return P.compact("V"); }

/// The columns of a vertex polytope in lexicographic order: a vertex set has no order.
Polytope aux_sorted(const Polytope &P) {
    const Tensor &V = P.V();
    const int64_t n = V.shape()[0], p = V.shape()[1];
    const std::vector<double> x = V.data();
    std::vector<int64_t> order(static_cast<std::size_t>(p));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        // the first coordinate that differs decides
        for (int64_t i = 0; i < n; ++i)
            if (x[i * p + a] != x[i * p + b]) return x[i * p + a] < x[i * p + b];
        return false;
    });
    std::vector<double> sorted;
    for (int64_t i = 0; i < n; ++i)
        for (const int64_t c : order) sorted.push_back(x[i * p + c]);
    return Polytope(Tensor::from_data(sorted, {n, p}));
}

/// The extreme vertices of CORA.cpp's result against those of the oracle's list (a vertex polytope
/// keeps every vertex it is given); an oracle result with error boxes has none to compare.
void aux_check_vertices(const std::string &what, const Polytope &mine,
                        const lean::Polytope &theirs) {
    if (!theirs.exact()) return;
    test::check(oracle::same_bits(what, aux_sorted(aux_extreme(mine)),
                                  lean::Polytope(aux_sorted(aux_extreme(theirs.gather())))),
                what + ": the extreme vertices of the oracle's list");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("polytope.origin", [](oracle::Random &) {
        oracle::check(Polytope::origin(3), lean::Polytope::origin(3));
    });
    oracle::run("polytope.isBounded", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 6));
        test::check(P.is_bounded() == lean::Polytope(P).is_bounded(), "isBounded: the Boolean");
    });
    oracle::run("polytope.isFullDim", [](oracle::Random &rnd) {
        const Polytope full(aux_sphere(rnd, 3, 8));
        test::check(full.is_full_dim() == lean::Polytope(full).is_full_dim(), "isFullDim: a ball");
        // a flat set: the points of a plane
        const Tensor flat = Tensor::concat({aux_sphere(rnd, 2, 6), Tensor::zeros({1, 6})}, 0);
        const Polytope plane(flat);
        test::check(plane.is_full_dim() == lean::Polytope(plane).is_full_dim(),
                    "isFullDim: a plane");
    });
    oracle::run("polytope.isequal", [](oracle::Random &rnd) {
        const Polytope a = aux_halfspaces(rnd, 3, 2), b = aux_halfspaces(rnd, 3, 2);
        test::check(a.isequal(a) == lean::Polytope(a).isequal(a), "isequal: itself");
        // the oracle never says true for sets that differ
        if (lean::Polytope(a).isequal(b)) test::check(a.isequal(b), "isequal: equal sets");
    });
    oracle::run("polytope.representsa", [](oracle::Random &rnd) {
        const Polytope P = aux_halfspaces(rnd, 3, 2);
        test::check(P.representsa("fullspace") == lean::Polytope(P).represents_fullspace(),
                    "representsa: fullspace, a bounded set");
        const Polytope all = Polytope::inf(3);
        test::check(all.representsa("fullspace") == lean::Polytope(all).represents_fullspace(),
                    "representsa: fullspace, R^3");
    });
    // the maps of the halfspace form
    oracle::run("polytope.minus", [](oracle::Random &rnd) {
        const Polytope P = aux_halfspaces(rnd, 3, 2);
        const Tensor v = rnd.vector(3);
        oracle::check(P - v, lean::Polytope(P).minus(v));
    });
    oracle::run("polytope.lift", [](oracle::Random &rnd) {
        const Polytope P = aux_halfspaces(rnd, 3, 2);
        const std::vector<int64_t> dims{4, 0, 2};
        oracle::check(P.lift(5, dims), lean::Polytope(P).lift(5, dims));
    });
    // the placement of the coordinates in a larger space, in both forms
    oracle::run("polytope.projectHighDim", [](oracle::Random &rnd) {
        const Polytope P = aux_halfspaces(rnd, 3, 2);
        const std::vector<int64_t> dims{3, 0, 4};
        oracle::check(P.project_high_dim(5, dims), lean::Polytope(P).project_high_dim(5, dims));
    });
    oracle::run("polytope.projectHighDim", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 6));
        const std::vector<int64_t> dims{3, 0, 4};
        aux_check_vertices("projectHighDim", P.project_high_dim(5, dims),
                           lean::Polytope(P).project_high_dim(5, dims));
    });
    // the vertex form: maps and sums list every vertex, so the extreme ones are compared
    oracle::run("polytope.mtimes", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 6));
        const Tensor M = rnd.matrix(2, 3);
        aux_check_vertices("mtimes", M * P, lean::Polytope(P).linear_map(M));
    });
    oracle::run("polytope.plus", [](oracle::Random &rnd) {
        const Polytope a(aux_sphere(rnd, 3, 5)), b(aux_sphere(rnd, 3, 5, 0.5));
        aux_check_vertices("plus", a + b, lean::Polytope(a).plus(b));
    });
    oracle::run("polytope.or", [](oracle::Random &rnd) {
        const Polytope a(aux_sphere(rnd, 3, 5)), b(aux_sphere(rnd, 3, 5, 1.0));
        aux_check_vertices("or", a.unite(b), lean::Polytope(a).unite(b));
    });
    oracle::run("polytope.project", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 4, 8));
        const std::vector<int64_t> dims{2, 0};
        aux_check_vertices("project", P.project(dims), lean::Polytope(P).project(dims));
    });
    // the upper bound of the support function of a vertex set
    oracle::run("polytope.supportFunc", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 7));
        const Tensor dir = rnd.direction(3);
        oracle::check(P.support_function(dir).data()[0], lean::Polytope(P).support_function(dir));
    });
    oracle::run("polytope.isIntersecting", [](oracle::Random &rnd) {
        const Polytope P = aux_halfspaces(rnd, 3, 2);
        const Tensor c = rnd.vector(3) * 0.0;
        const Interval near(c - 0.1, c + 0.1), far(c + 10.0, c + 11.0);
        // disjointness proven by the oracle holds for CORA.cpp
        oracle::check(P.is_intersecting(near), lean::Polytope(P).is_disjoint(near));
        oracle::check(P.is_intersecting(far), lean::Polytope(P).is_disjoint(far));
    });
    return test::finish("lean polytope maps");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
