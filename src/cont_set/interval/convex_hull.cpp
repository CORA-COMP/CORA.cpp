// convex_hull - the convex hull of two boxes, as CORA's interval.convHull_
//
// Syntax:   Interval R = I.convex_hull(I2);
// Inputs:   I2 - interval of the dimension of I
// Outputs:  R - [min(inf, inf2), max(sup, sup2)], CORA's or; an empty operand returns the other;
//           nothing rounds
// See also: enclose, Interval::intersect
//
// CORA: contSet/@interval/convHull_.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::convex_hull(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "convex_hull: the intervals differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(I2.dim());
    });
    if (is_empty()) return I2;
    if (I2.is_empty()) return *this;
    return {inf_.minimum(I2.inf()), sup_.maximum(I2.sup())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
