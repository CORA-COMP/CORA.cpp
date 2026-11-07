// plus - the Minkowski sum of a capsule with a set or a vector, as CORA's capsule.plus
//
// The sum of two capsules is enclosed: the centers add, the longer generator stays (ours on a tie)
// and the length of the shorter one joins the radius. A vector shifts the center; a box or a
// zonotope shifts it by the center of its box and widens the radius by the box's. An empty operand
// gives the empty capsule.
//
// Syntax:   Cres = C.plus(S);   Cres = C.plus(v);   Cres = C.plus(I);   Cres = C.plus(Z);
// Inputs:   S - capsule of the dimension of C
//           v - column (..., n, 1)
//           I - box
//           Z - zonotope
// Outputs:  Cres - a capsule that contains C + S
// See also: convex_hull, Zonotope::plus
//
// CORA: contSet/@capsule/plus.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Capsule Capsule::plus(const Capsule &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the capsules differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    // the squared lengths order the generators as the lengths do; on a tie ours stays
    const Tensor mine = g_.square().sum(-2, true), theirs = S.g_.square().sum(-2, true);
    const Tensor keep_ours = theirs.less_equal(mine);
    const Tensor absorbed = Tensor::where(keep_ours, priv_norm(S.g_), priv_norm(g_));
    const Tensor r = r_.add(S.r_).add(absorbed);
    // either operand empty: the sum is empty, marked by the radius -1
    const Tensor empty = priv_empty_mask(r_).maximum(priv_empty_mask(S.r_));
    return {Unchecked{}, c_.add(S.c_), Tensor::where(keep_ours, g_, S.g_),
            Tensor::where(empty, r_.full_like(r_.shape(), -1.0), r)};
}

Capsule Capsule::plus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(is_column(v, dim()), Error::dimension_mismatch, [&] {
        return "plus: the vector must be a column of the dimension " + std::to_string(dim()) +
               ", got the shape " + shape_text(v.shape());
    });
    return {Unchecked{}, c_.add(v), g_, r_};
}

Capsule Capsule::plus(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the box differs in dimension, " + std::to_string(I.dim()) + " and " +
               std::to_string(dim());
    });
    // the box widens the radius by that of the ball around it, the 2-norm of its radius
    const Tensor r = r_.add(I.rad().vector_norm(Tensor::Norm::two, -2, true));
    return {Unchecked{}, c_.add(I.center()), g_, Tensor::where(priv_empty_mask(r_), r_, r)};
}

Capsule Capsule::plus(const Zonotope &Z) const { return plus(Z.interval()); }

Set Capsule::translate_impl(const Tensor &v) const { return plus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
