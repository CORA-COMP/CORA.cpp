// unite - the smallest box around a box and a box or a point, as CORA's interval.or
//
// Syntax:   Interval R = I.unite(I2);   Interval R = I.unite(p);
// Inputs:   I2 - interval of the dimension of I
//           p - a point, a column (..., n, 1)
// Outputs:  R - [min(inf, inf2), max(sup, sup2)]; an empty operand returns the other
// See also: convex_hull, max
//
// CORA: contSet/@interval/or.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::unite(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "unite: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    // CORA's convHull_ ends in or: one code path
    return convex_hull(I2);
}

Interval Interval::unite(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        return "unite: the point must be a column of the dimension " + std::to_string(dim()) +
               " of the interval, got the shape " + shape_text(p.shape());
    });
    return convex_hull(Interval(p, p));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
