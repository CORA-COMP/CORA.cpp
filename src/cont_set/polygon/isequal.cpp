// isequal - whether two polygons are the same region, as CORA's isequal
//
// Every piece of each lies in a piece of the other, all its vertices in a vertex triangle
// (CORALean's Polygon.isequalPieces). true means the same region; a piece covered only by
// several pieces of the other gives false. CORA compares the boundary vertices.
//
// Syntax:   bool b = pgon.isequal(S);
// Inputs:   S - a polygon
//           tol - only 0 (default: unset)
// Outputs:  b - whether the pieces cover each other
// See also: contains
//
// CORA: contSet/@polygon/isequal.m
// CORA authors: Mark Wetzlinger
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

/// Whether every piece of A lies in a piece of B.
bool aux_covered(const std::vector<Piece> &A, const std::vector<Piece> &B) {
    for (const Piece &p : A) {
        bool in = false;
        for (const Piece &q : B) in = in || priv_piece_subset(p, q);
        if (!in) return false;
    }
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Polygon::isequal(const Polygon &S, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(!tol || *tol == 0, Error::not_supported,
               [] { return std::string("isequal: only tol = 0 is supported, as in CORALean"); });
    const std::vector<Piece> A = priv_hosts(*this), B = priv_hosts(S);
    return aux_covered(A, B) && aux_covered(B, A);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
