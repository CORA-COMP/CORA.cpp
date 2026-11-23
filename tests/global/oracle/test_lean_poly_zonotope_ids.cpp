// test_lean_poly_zonotope_ids - the identifier and predicate ops of PolyZonotope against CORALean
//
// The ops of `polyZonotope.`: replaceId, restoreId, rmId, onlyId, noIndep, lift, getSubset and
// resolve (the factors at numbers, exponents up to 3), isBounded, isFullDim, isequal, representsa
// (point). Random sets with equal and constant columns; every column of the result (generators,
// exponents, identifiers) is compared in its order.
//
// isequal: the oracle's is structural and in order, a `true` is a proof, CORA.cpp's compacts the
// sets and compares up to order. The test restricts to the sets where the two agree: a set with
// itself (the same order) and with sets that differ in the center or in the columns.
//
// ORACLE: contSet/@polyZonotope/replaceId.m
// ORACLE: contSet/@polyZonotope/restoreId.m
// ORACLE: contSet/@polyZonotope/rmId.m
// ORACLE: contSet/@polyZonotope/onlyId.m
// ORACLE: contSet/@polyZonotope/noIndep.m
// ORACLE: contSet/@polyZonotope/lift_.m
// ORACLE: contSet/@polyZonotope/getSubset.m
// ORACLE: contSet/@polyZonotope/resolve.m
// ORACLE: contSet/@polyZonotope/isBounded.m
// ORACLE: contSet/@polyZonotope/isFullDim.m
// ORACLE: contSet/@polyZonotope/isequal.m
// ORACLE: contSet/@polyZonotope/representsa_.m
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

/// A vector (n) of n numbers in [-1, 1).
Tensor aux_values(oracle::Random &rnd, std::size_t n) {
    std::vector<double> v(n);
    for (double &x : v) x = rnd.uniform(-1, 1);
    return Tensor::from_data(v, {int64_t(n)});
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // identifiers renamed: two factors to one (they merge), then all of them
    oracle::run("polyZonotope.replaceId", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const std::vector<int64_t> &id = P.ids();
            oracle::check("merge", P.replace_id({id[0], id[1]}, {id[1], id[1]}),
                          L.replace_id({id[0], id[1]}, {id[1], id[1]}));
            oracle::check("rename", P.replace_id({id[0]}, {41}), L.replace_id({id[0]}, {41}));
            std::vector<int64_t> all(id.size());
            for (std::size_t i = 0; i < id.size(); ++i) all[i] = i < 2 ? 50 : 60 + int64_t(i);
            oracle::check("all", P.replace_id(all), L.replace_id(all));
        }
    });

    // identifiers the set lacks added; one it has is skipped
    oracle::run("polyZonotope.restoreId", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const std::vector<int64_t> ids = {P.ids()[0], 41, 40, 2};
            oracle::check(P.restore_id(ids), lean::PolyZonotope(P).restore_id(ids));
        }
    });

    // factors fixed to 1, equal columns merged
    oracle::run("polyZonotope.rmId", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const std::vector<int64_t> &id = P.ids();
            oracle::check("one", P.rm_id({id[1]}), L.rm_id({id[1]}));
            oracle::check("two", P.rm_id({id[0], id.back()}), L.rm_id({id[0], id.back()}));
            oracle::check("none", P.rm_id({}), L.rm_id({}));
        }
    });

    // the part that depends on the factors alone, and the rest
    oracle::run("polyZonotope.onlyId", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const std::vector<int64_t> &id = P.ids();
            oracle::check("one", P.only_id({id[1]}), L.only_id({id[1]}));
            oracle::check("two", P.only_id({id[0], id.back()}), L.only_id({id[0], id.back()}));
        }
    });

    oracle::run("polyZonotope.noIndep", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check(P.no_indep(), lean::PolyZonotope(P).no_indep());
        }
    });

    // the coordinates permuted
    oracle::run("polyZonotope.lift", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            std::vector<int64_t> dims(std::size_t(s.n));
            for (int64_t i = 0; i < s.n; ++i) dims[std::size_t(i)] = s.n - 1 - i;
            oracle::check(P.lift(s.n, dims), lean::PolyZonotope(P).lift(s.n, dims));
        }
    });

    // the factors at numbers, one after the other: exponents up to 3 multiply the value up
    oracle::run("polyZonotope.getSubset", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const std::vector<int64_t> &id = P.ids();
            const Tensor one = aux_values(rnd, 1), two = aux_values(rnd, 2);
            oracle::check("one", P.get_subset({id[1]}, one), L.get_subset({id[1]}, one));
            oracle::check("two", P.get_subset({id[0], id[1]}, two),
                          L.get_subset({id[0], id[1]}, two));
            const Tensor all = aux_values(rnd, id.size());
            oracle::check("all", P.get_subset(id, all), L.get_subset(id, all));
        }
    });

    oracle::run("polyZonotope.resolve", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            const lean::PolyZonotope L(P);
            const std::vector<int64_t> &id = P.ids();
            const Tensor all = aux_values(rnd, id.size());
            oracle::check("all", P.resolve(all), L.resolve(all));
            // some of the factors, not in ascending order
            const std::vector<int64_t> some = {id[1], id[0]};
            const Tensor x = aux_values(rnd, 2);
            oracle::check("ids", P.resolve(x, some), L.resolve(x, some));
        }
    });

    oracle::run("polyZonotope.isBounded", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check(P.is_bounded(), lean::PolyZonotope(P).is_bounded());
        }
    });

    // a set with enough generators, and a segment (one generator)
    oracle::run("polyZonotope.isFullDim", [](oracle::Random &rnd) {
        for (const PolyDims &s : {PolyDims{2, 6, 3, 2}, {3, 9, 4, 3}, {2, 1, 2, 0}, {3, 2, 2, 0}}) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check(P.is_full_dim(), lean::PolyZonotope(P).is_full_dim());
        }
    });

    // a set with itself, with another center and with another set
    oracle::run("polyZonotope.isequal", [](oracle::Random &rnd) {
        for (const PolyDims &s : kDims) {
            oracle::PolyColumns r = oracle::poly_columns(rnd, s);
            const PolyZonotope P = oracle::poly_of(r);
            const lean::PolyZonotope L(P);
            oracle::check("same", P.isequal(P), L.isequal(L));
            r.c = r.c + r.c.ones_like();
            const PolyZonotope shifted = oracle::poly_of(r);
            oracle::check("center", P.isequal(shifted), L.isequal(lean::PolyZonotope(shifted)));
            const PolyZonotope other = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check("other", P.isequal(other), L.isequal(lean::PolyZonotope(other)));
        }
    });

    // a point (no generator at all), a point with a zero center column and a set with generators
    oracle::run("polyZonotope.representsa", [](oracle::Random &rnd) {
        for (const PolyDims &s : {PolyDims{2, 0, 2, 0}, {3, 0, 3, 0}, {2, 3, 2, 1}, {3, 6, 3, 2}}) {
            const PolyZonotope P = oracle::poly_of(oracle::poly_columns(rnd, s));
            oracle::check(P.representsa("point"), lean::PolyZonotope(P).representsa_point());
        }
    });

    return test::finish("lean poly_zonotope ids");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
