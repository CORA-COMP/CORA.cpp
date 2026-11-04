// convex_hull - a zonotope that encloses the convex hull of a zonotope and another set
//
// Syntax:   Zres = Z.convex_hull(Z2);   Zres = Z.convex_hull(I);   Zres = Z.convex_hull(p);
// Inputs:   Z2, I, p - a zonotope, a box or a point (n, 1) of the dimension of Z
// Outputs:  Zres - Z.enclose(zonotope(S)): an outer enclosure, coarser than the hull
// See also: enclose, linear_combination
//
// CORA: contSet/@zonotope/convHull_.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::convex_hull(const Zonotope &Z2) const {
    detail::Entry entry;
    CORA_CHECK(Z2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "convex_hull: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z2.dim());
    });
    return enclose(Z2);
}

Zonotope Zonotope::convex_hull(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.dim() == dim() && I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "convex_hull: the interval must be a box of the dimension " + std::to_string(dim()) +
               " of the zonotope, got bounds of the shape " + shape_text(I.inf().shape());
    });
    // a box is the zonotope of its axis generators
    return enclose(I.zonotope());
}

Zonotope Zonotope::convex_hull(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        return "convex_hull: the point must be a column of the dimension " + std::to_string(dim()) +
               " of the zonotope, got the shape " + shape_text(p.shape());
    });
    // a point is the zonotope without generators
    return enclose(priv_zonotope_of_point(p));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
