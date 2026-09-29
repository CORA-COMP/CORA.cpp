// contains - point containment in a box, as CORA's interval.contains
//
// Syntax:   res = I.contains(p);
// Inputs:   p - point (..., n, 1)
// Outputs:  res - true if p is in the box (boundary included); for a batch, if all points are
// See also: Specification::check

#include "contSet/interval/interval.h"

namespace cora::ct {

namespace {
Tensor aux_distanceOutside(const Interval &I, const Tensor &p);
} // namespace

bool Interval::contains(const Tensor &p) const {
    for (const double outside : aux_distanceOutside(*this, p).data())
        if (outside > 1e-12) return false;  // tolerance for rounding on the boundary
    return true;
}

// --------------------------- auxiliary functions --------------------------------

namespace {

/// How far p is outside the box, summed over the dimensions: (..., 1, 1), zero inside.
Tensor aux_distanceOutside(const Interval &I, const Tensor &p) {
    return ((I.inf - p).pos() + (p - I.sup).pos()).transpose().sumLast();
}

} // namespace

} // namespace cora::ct
