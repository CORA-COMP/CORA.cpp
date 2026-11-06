// convex_hull - a capsule that encloses two capsules, as CORA's capsule.convHull_
//
// Syntax:   Cres = C.convex_hull(S);
// Inputs:   S - capsule of the dimension of C
// Outputs:  Cres - the axis between the two centers, c_ = (c + S.c)/2 and g_ = (c - S.c)/2, with
//           the radius max(|g| + r, |S.g| + S.r): every point of each capsule is within that of
//           its center. The hull with an empty capsule is the other one (CORA stops with an
//           unassigned output there)
// See also: enclose, Interval::convex_hull
//
// CORA: contSet/@capsule/convHull_.m

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Capsule Capsule::convex_hull(const Capsule &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "convex_hull: the capsules differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    // g_ is half the difference of the centers, as CORALean computes it, not c - c_
    const Tensor c = c_.add(S.c_) / 2.0, g = c_.subtract(S.c_) / 2.0;
    const Tensor reach = priv_norm(g_).add(r_), reach2 = priv_norm(S.g_).add(S.r_);
    const Tensor r = reach.maximum(reach2);
    // an empty operand adds nothing: the hull is the other capsule
    const Tensor mine = priv_empty_mask(r_), theirs = priv_empty_mask(S.r_);
    const auto pick = [&](const Tensor &hull, const Tensor &own, const Tensor &other) {
        return Tensor::where(theirs, own, Tensor::where(mine, other, hull));
    };
    return {Unchecked{}, pick(c, c_, S.c_), pick(g, g_, S.g_), pick(r, r_, S.r_)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
