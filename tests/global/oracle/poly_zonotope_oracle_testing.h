// poly_zonotope_oracle_testing - the random polynomial zonotopes of the oracle tests
//
// Syntax:   const PolyColumns r = poly_columns(rnd, {n, h, p, q});   PolyZonotope P = poly_of(r);
//           lean::PolyZonotope L(P);   std::vector<int64_t> id = poly_ids(p);
// Sets:     n dimensions, h dependent generators on p factors with the exponents 0..3 (a constant
//           column and equal columns happen), q independent generators; the constructor of CORA
//           merges the equal columns, so the set is what CORA.cpp and the oracle both read
// See also: oracle_testing.h
//
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/global/oracle/poly_zonotope.h"
#include "oracle_testing.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test::oracle {

/// The shape of a random set: n dimensions, h dependent generators over p factors, q independent.
struct PolyDims {
    int64_t n, h, p, q;
};

/// The columns of a random set before the constructor, CORA's (n, h) generators and the exponents.
struct PolyColumns {
    cora::Tensor c, G, GI;
    cora::Exponents E;
    std::vector<int64_t> id;
};

/// The identifiers of p factors: 3, 5, 7, ..., each raised by shift.
inline std::vector<int64_t> poly_ids(int64_t p, int64_t shift = 0) {
    std::vector<int64_t> id(static_cast<std::size_t>(p));
    for (int64_t i = 0; i < p; ++i) id[std::size_t(i)] = 2 * i + 3 + shift;
    return id;
}

/// The columns of a random set of the shape over the identifiers id (default: poly_ids(p)).
inline PolyColumns poly_columns(Random &rnd, const PolyDims &s, std::vector<int64_t> id = {}) {
    cora::Exponents E;
    for (int64_t l = 0; l < s.h; ++l) {
        std::vector<int> col(std::size_t(s.p), 0);
        for (int &e : col) e = static_cast<int>(rnd.uniform(0, 4));
        E.push_back(col);
    }
    if (id.empty()) id = poly_ids(s.p);
    return {rnd.vector(s.n), rnd.matrix(s.n, s.h), rnd.matrix(s.n, s.q), E, std::move(id)};
}

/// The set of the columns, as CORA.cpp stores it (rows).
inline cora::PolyZonotope poly_of(const PolyColumns &r) {
    return {r.c.squeeze(-1), r.G.matrix_transpose(), r.GI.matrix_transpose(), r.E, r.id};
}

} // namespace test::oracle

// ---------------------------------------  END OF CODE  ---------------------------------------- //
