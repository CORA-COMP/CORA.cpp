// intersect - the intersection of two boxes, as CORA's interval.and_
//
// Syntax:   Interval R = I.intersect(I2);
// Inputs:   I2 - interval of the dimension of I
// Outputs:  R - [max(inf, inf2), min(sup, sup2)] if the gap of the bounds is at most eps (for a gap
//           in (0, eps] the bounds are ordered); else the empty interval (bounds without columns).
//           For a batch all members must agree on whether they are empty
// See also: plus, contains
//
// CORA: contSet/@interval/and_.m

#include "cont_set/interval/private/priv_empty.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::intersect(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "intersect: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    if (is_empty() || I2.is_empty()) return priv_empty(is_empty() ? inf_ : I2.inf());
    const Tensor lb = inf_.maximum(I2.inf()), ub = sup_.minimum(I2.sup());
    const double kEps = std::numeric_limits<double>::epsilon();
    // 1 where a member's boxes meet, 0 where they are apart by more than eps
    const Tensor meet =
        (lb - ub).less_equal(lb.full_like(lb.shape(), kEps)).all(-2, true).all(-1, true);
    if (meet.max().item() == 0) return priv_empty(lb);
    CORA_CHECK(meet.min().item() != 0, Error::not_supported, [] {
        return std::string("intersect: the members of a batch must all meet or all be disjoint");
    });
    return {lb.minimum(ub), lb.maximum(ub)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
