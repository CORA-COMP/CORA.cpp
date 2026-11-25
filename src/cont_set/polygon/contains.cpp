// contains - whether a polygon holds points, as CORA's contains_
//
// Each point lies in a vertex triangle of some piece (CORALean's Polygon.containsPoint: a left
// turn a, b, c and the point on the inner side of each edge, at every ordered triple).
// A boundary point counts. CORA's tol and the containment of sets are not ported.
//
// Syntax:   bool b = pgon.contains(p);
// Inputs:   p - points (..., 2)
//           tol - only 0 (default: unset)
// Outputs:  b - whether all points lie in the polygon
// See also: is_intersecting, priv_piece_holds
//
// CORA: contSet/@polygon/contains_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Polygon::contains(const Tensor &p, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(p.defined() && p.ndim() >= 1 && p.size(-1) == 2, Error::dimension_mismatch,
               [&] { return std::string("contains: the points must be (..., 2)"); });
    CORA_CHECK(!tol || *tol == 0, Error::not_supported,
               [] { return std::string("contains: only tol = 0 is supported, as in CORALean"); });
    const std::vector<Piece> hosts = priv_hosts(*this);
    const std::vector<double> x = p.data();
    // every point in some piece
    for (std::size_t r = 0; 2 * r < x.size(); ++r) {
        bool held = false;
        for (const Piece &v : hosts) held = held || priv_piece_holds(v, {x[2 * r], x[2 * r + 1]});
        if (!held) return false;
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
