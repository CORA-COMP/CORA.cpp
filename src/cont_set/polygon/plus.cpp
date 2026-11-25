// plus - the Minkowski sum of a polygon with a polygon or a point, as CORA's plus and minus
//
// With a polygon: every pair of pieces, this one's outer, summed vertex by vertex, k1 k2
// vertices each (CORALean's Polygon.plus, VPolytope.plus). With a point: every vertex shifted,
// which is minus(-p) (CORALean's Polygon.minus, the sum with the piece {-p}).
//
// Syntax:   Polygon R = pgon.plus(S);   Polygon R = pgon.plus(p);   Set R = pgon.minus(p);
// Inputs:   S - a polygon
//           p - a point (2)
// Outputs:  R - the sum
// See also: unite, linear_map
//
// CORA: contSet/@polygon/plus.m
// CORA: contSet/@polygon/minus.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The pieces shifted by v (2).
std::vector<Polytope> aux_shifted(const std::vector<Polytope> &pieces, const Tensor &v) {
    std::vector<Polytope> out;
    out.reserve(pieces.size());
    for (const Polytope &P : pieces) out.emplace_back(P.V() + v.reshape({1, 2}));
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polygon Polygon::plus(const Polygon &S) const {
    detail::Entry entry;
    std::vector<Polytope> pieces;
    for (const Polytope &P : pieces_)
        for (const Polytope &Q : S.pieces_) {
            // the vertex l = i k2 + j is p_i + q_j
            const int64_t k1 = P.V().size(0), k2 = Q.V().size(0);
            const Tensor sums = P.V().reshape({k1, 1, 2}) + Q.V().reshape({1, k2, 2});
            pieces.emplace_back(sums.reshape({k1 * k2, 2}));
        }
    return priv_polygon(std::move(pieces), like_);
}

Polygon Polygon::plus(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(p.defined() && p.numel() == 2, Error::dimension_mismatch,
               [&] { return std::string("plus: the point must have two entries"); });
    return priv_polygon(aux_shifted(pieces_, p), like_);
}

Set Polygon::translate_impl(const Tensor &v) const {
    return priv_polygon(aux_shifted(pieces_, v), like_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
