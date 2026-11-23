// test_lean_poly_zonotope_maps - the combining and mapping ops of PolyZonotope, against the oracle
//
// The ops of `polyZonotope.`: sum, restructure, jacobian, linComb, convHull, matMap, cubMap,
// partZonotope and outerApprox. Random sets with equal and constant columns; every column of the
// result (generators, exponents, identifiers) is compared in its order. matMap and cubMap sum
// their dot products from the last term to the first, as CORALean's matMulDown and
// cubContractDown, which the sizes k, n >= 3 of the sums make visible.
//
// ORACLE: contSet/@polyZonotope/sum.m
// ORACLE: contSet/@polyZonotope/restructure.m
// ORACLE: contSet/@polyZonotope/jacobian.m
// ORACLE: contSet/@polyZonotope/linComb.m
// ORACLE: contSet/@polyZonotope/convHull_.m
// ORACLE: contSet/@polyZonotope/matMap.m
// ORACLE: contSet/@polyZonotope/cubMap.m
// ORACLE: contSet/@polyZonotope/partZonotope.m
// ORACLE: contSet/@polyZonotope/outerApprox.m
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;
using oracle::PolyDims;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The shapes the ops are tried on.
const std::vector<PolyDims> kDims = {{2, 3, 2, 0}, {2, 6, 3, 2}, {3, 9, 4, 3}, {4, 12, 4, 2}};

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // the sum of sets over the identifiers of the first
    oracle::run("polyZonotope.sum", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            std::vector<PolyZonotope> list;
            for (const int64_t h : {s.h / 2 + 1, s.h})
                list.push_back(
                    oracle::poly_of(oracle::poly_columns(rnd, {s.n, h, s.p, s.q}, P.ids())));
            std::vector<lean::PolyZonotope> lean_list;
            lean_list.reserve(list.size());
            for (const PolyZonotope &Q : list) lean_list.emplace_back(Q);
            oracle::check("two", P.sum(list), lean::PolyZonotope(P).sum(lean_list));
            oracle::check("one", P.sum({list[0]}), lean::PolyZonotope(P).sum({lean_list[0]}));
        }
    });

    // the independent part boxed and made dependent
    oracle::run("polyZonotope.restructure", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check(P.restructure(), lean::PolyZonotope(P).restructure());
        }
    });

    // the derivative along each factor
    oracle::run("polyZonotope.jacobian", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            for (const int64_t id : P.ids())
                oracle::check("factor " + std::to_string(id),
                              P.jacobian(std::vector<int64_t>{id})[0], L.jacobian(id));
        }
    });

    // two sets over other identifiers, and over the same ones
    oracle::run("polyZonotope.linComb", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            // the second set on other identifiers (apart) and on the same ones (shared)
            const PolyZonotope Q =
                oracle::poly_of(oracle::poly_columns(rnd, s, oracle::poly_ids(s.p, 1)));
            const PolyZonotope R = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check("apart", P.linear_combination(Q),
                          L.linear_combination(lean::PolyZonotope(Q)));
            oracle::check("shared", P.linear_combination(R),
                          L.linear_combination(lean::PolyZonotope(R)));
        }
    });

    oracle::run("polyZonotope.convHull", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const PolyZonotope Q =
                oracle::poly_of(oracle::poly_columns(rnd, s, oracle::poly_ids(s.p, 1)));
            const PolyZonotope R = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check("hull apart", P.convex_hull(Q), L.convex_hull(lean::PolyZonotope(Q)));
            oracle::check("hull shared", P.convex_hull(R), L.convex_hull(lean::PolyZonotope(R)));
        }
    });

    // the product of an n x k and a k x m matrix of sets over overlapping identifiers
    oracle::run("polyZonotope.matMap", [](oracle::Random &rnd) {
        for (const auto &[n, k, m] : {std::tuple{2, 2, 2}, {2, 3, 2}, {1, 4, 1}, {3, 3, 2}}) {
            const PolyZonotope A =
                oracle::poly_of(oracle::poly_columns(rnd, {n * k, 3, 3, 2}, {3, 5, 7}));
            const PolyZonotope B =
                oracle::poly_of(oracle::poly_columns(rnd, {k * m, 4, 3, 1}, {5, 7, 9}));
            oracle::check(A.matrix_map(B, n, k, m),
                          lean::PolyZonotope(A).matrix_map(lean::PolyZonotope(B), n, k, m));
        }
    });

    // the cubic map: the triples of generators, the sums over k of the cubic form
    oracle::run("polyZonotope.cubMap", [](oracle::Random &rnd) {
        for (const PolyDims &s : {PolyDims{2, 2, 2, 1}, {3, 2, 2, 0}, {4, 1, 1, 0}}) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            std::vector<std::vector<Tensor>> T;
            for (int64_t i = 0; i < 2; ++i) {
                T.emplace_back();
                for (int64_t k = 0; k < s.n; ++k) T.back().push_back(rnd.matrix(s.n, s.n));
            }
            oracle::check(P.cubic_map(T), lean::PolyZonotope(P).cubic_map(T));
        }
    });

    // the generators grouped by the exponents of some factors
    oracle::run("polyZonotope.partZonotope", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const std::vector<int64_t> &id = P.ids();
            for (const std::vector<int64_t> &ids :
                 {std::vector<int64_t>{}, {id[1]}, {id[0], id[1]}})
                oracle::check("ids " + std::to_string(ids.size()), P.part_zonotope(ids),
                              L.part_zonotope(ids));
        }
    });

    // the small generators boxed, at tolerances that take none, some and all of them
    oracle::run("polyZonotope.outerApprox", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const std::vector<int64_t> &id = P.ids();
            for (const double tol : {0.0, 0.3, 0.9, 5.0})
                for (const std::vector<int64_t> &ids : {std::vector<int64_t>{}, {id[0]}})
                    oracle::check("tol " + std::to_string(tol), P.outer_approx(tol, ids),
                                  L.outer_approx(tol, ids));
        }
    });

    return test::finish("lean poly_zonotope maps");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
