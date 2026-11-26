// is_full_dim - whether a polygon has an interior, as CORA's isFullDim
//
// Some piece has three distinct vertices whose orientation is nonzero (CORALean's
// Polygon.isFullDim, the six products of orientBounds). tol is not used, as in CORALean.
//
// Syntax:   bool b = pgon.is_full_dim();
// Inputs:   tol - not used
// Outputs:  b - whether some piece has an area
// See also: volume
//
// CORA: contSet/@polygon/isFullDim.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Polygon::is_full_dim_impl(Tol) const {
    for (const Piece &p : priv_hosts(*this))
        for (const Point &a : p)
            for (const Point &b : p)
                for (const Point &c : p) {
                    const double o = priv_orient(a, b, c);
                    if (priv_distinct(a, b, c) && (0.0 < o || o < 0.0)) return true;
                }
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
