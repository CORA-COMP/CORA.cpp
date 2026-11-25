// volume - the area of a polygon, as CORA's volume_
//
// The pieces are made disjoint (every piece minus the earlier ones, by clipping) and their
// areas summed (shoelace). CORALean bounds the area by the box's, so this is CORA.cpp's.
//
// Syntax:   double A = pgon.volume();
// Inputs:   -
// Outputs:  A - the area of the region
// See also: center, priv_disjoint
//
// CORA: contSet/@polygon/volume_.m
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

double Polygon::volume() const {
    detail::Entry entry;
    double area = 0;
    for (const Piece &part : priv_disjoint(priv_hosts(*this))) area += priv_area(part);
    return area;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
