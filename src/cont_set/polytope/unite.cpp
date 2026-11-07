// unite - an enclosure of the union of a polytope and a set, as CORA's polytope.or
//
// CORA computes it as the convex hull (convHull_), which is the smallest convex set around the
// union; so does this.
//
// Syntax:   Pres = P.unite(S);   Pres = P.unite(I);   Pres = P.unite(p);
// Inputs:   S, I - a polytope or a box of the dimension of P
//           p - a point (n, 1)
// Outputs:  Pres - the convex hull of both, as Polytope::convex_hull
// See also: Polytope::convex_hull, Polytope::enclose
//
// CORA: contSet/@polytope/or.m
// CORA authors: Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::unite(const Polytope &S) const { return convex_hull(S); }

Polytope Polytope::unite(const Interval &I) const { return convex_hull(I); }

Polytope Polytope::unite(const Tensor &p) const { return convex_hull(p); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
