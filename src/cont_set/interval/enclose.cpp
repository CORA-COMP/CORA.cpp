// enclose - a box around two boxes, as CORA's interval.enclose
//
// Syntax:   Interval R = I.enclose(I2);
// Inputs:   I2 - interval of the dimension of I
// Outputs:  R - the convex hull of I and I2, as CORA's enclose calls convHull
// See also: convex_hull
//
// CORA: contSet/@interval/enclose.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::enclose(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "enclose: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    return convex_hull(I2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
