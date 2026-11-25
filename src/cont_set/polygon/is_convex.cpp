// is_convex - whether a polygon is convex, as CORA's isConvex
//
// No piece, or one piece that holds every piece (CORALean's Polygon.isConvex): then the
// region is that piece. true means convex; a convex union of several pieces gives false.
//
// Syntax:   bool b = pgon.is_convex();
// Inputs:   -
// Outputs:  b - whether the polygon is one convex piece
// See also: isequal, split_into_convex_sets
//
// CORA: contSet/@polygon/isConvex.m
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

bool Polygon::is_convex() const {
    detail::Entry entry;
    const std::vector<Piece> hosts = priv_hosts(*this);
    if (hosts.empty()) return true;
    for (const Piece &p : hosts) {
        bool all = true;
        for (const Piece &q : hosts) all = all && priv_piece_subset(q, p);
        if (all) return true;
    }
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
