// con_poly_zono_oracle_testing - random constrained polynomial zonotopes for the oracle tests
//
// A set of n dimensions with h dependent generators over p factors (exponents 0..2, the first
// column repeated so that compact has a merge), q independent generators, and m rows over r
// monomials; the values are normal draws, the identifiers 1..p.
//
// Syntax:   ConPolyZono S = test::oracle::con_poly_zono(rnd, {n, h, p, q, m, r});
// See also: oracle_testing.h, poly_zonotope_oracle_testing.h
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_poly_zono/con_poly_zono.h"
#include "oracle_testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test::oracle {

/// The shape of a random set: n dimensions, h dependent generators over p factors, q independent
/// ones, m rows over r monomials.
struct ConPolyDims {
    int64_t n, h, p, q, m, r;
};

/// Exponent columns of p entries in 0..2, the first column repeated last (h >= 2).
inline cora::Exponents random_exponents(Random &rnd, int64_t count, int64_t p) {
    cora::Exponents E;
    for (int64_t l = 0; l < count; ++l) {
        std::vector<int> col(std::size_t(p), 0);
        for (int &e : col) e = static_cast<int>(rnd.uniform(0, 3));
        E.push_back(col);
    }
    if (count >= 2) E.back() = E.front();
    return E;
}

/// A random constrained polynomial zonotope of the shape, identifiers 1..p.
inline cora::ConPolyZono con_poly_zono(Random &rnd, const ConPolyDims &s) {
    std::vector<int64_t> id(std::size_t(s.p));
    for (int64_t i = 0; i < s.p; ++i) id[std::size_t(i)] = i + 1;
    return {rnd.point(s.n),
            rnd.matrix(s.h, s.n),
            random_exponents(rnd, s.h, s.p),
            rnd.matrix(s.m, s.r),
            rnd.point(s.m),
            random_exponents(rnd, s.r, s.p),
            rnd.matrix(s.q, s.n),
            id};
}

} // namespace test::oracle

// ---------------------------------------  END OF CODE  ---------------------------------------- //
