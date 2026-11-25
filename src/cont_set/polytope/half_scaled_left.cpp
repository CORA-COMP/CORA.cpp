// half_scaled_left - a set's rows at a convex-hull weight, as CORALean's halfScaledLeft
//
// For the convex hull of two sets with the weight w in [-1, 1]: the rows A x <= b (1 + w) / 2 as
// [A, 0_k, -b/2] <= b/2 in the coordinates (x, y, w), y the k coordinates of the other set; at
// w = 1 the set, at w = -1 its cone A x <= 0. Equalities stay equalities (CORALean's float
// definition splits each in two inequalities). A polytope in vertex form is converted by
// constraints first (not a batch). The empty-set flag is not kept (w = -1 holds x = 0).
//
// Syntax:   Pres = P.half_scaled_left(k);
// Inputs:   P - polytope of dimension n
//           k - the coordinates of the other set, k >= 0
// Outputs:  Pres - a polytope of dimension n + k + 1
// See also: half_scaled_right, ConZonotope::convex_hull
//
// CORA.cpp only: CORALean's HPolytope.halfScaledLeft, the rows of conZonotope's convHull_
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::half_scaled_left(int64_t k) const {
    detail::Entry entry;
    CORA_CHECK(k >= 0, Error::wrong_value, [&] {
        return "half_scaled_left: the coordinates of the other set must be nonnegative, got " +
               std::to_string(k);
    });
    const Constraints H = priv_half_scaled(constraints(), k, true);
    // the set at w = -1 is a cone, which holds 0: no empty flag
    return Polytope(Parts{H.A, H.b, H.Ae, H.be, Tensor(), EmptyMask()});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
