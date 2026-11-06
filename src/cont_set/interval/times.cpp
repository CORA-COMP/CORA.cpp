// times - the entrywise product of a box with a box or with numbers, as CORA's interval.times
//
// Syntax:   Interval R = I.times(I2);   Interval R = I.times(F);
// Inputs:   I2 - a box
//           F - numbers (a 1 x 1 factor is a scalar);  the operands broadcast
// Outputs:  R - the hull of the four corner products (the numbers are a point box), a corner
//           0 * Inf taken as 0; nothing rounds outward
// See also: prod, rdivide, linear_map
//
// CORA: contSet/@interval/times.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_broadcastable.h"
#include "cont_set/interval/private/priv_times.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::times(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(priv_broadcastable(inf_.shape(), I2.inf().shape()), Error::dimension_mismatch, [&] {
        return "times: the shapes " + shape_text(inf_.shape()) + " and " +
               shape_text(I2.inf().shape()) + " do not broadcast";
    });
    const auto [lo, hi] = priv_times(inf_, sup_, I2.inf(), I2.sup());
    return {lo, hi};
}

Interval Interval::times(const Tensor &factor) const {
    detail::Entry entry;
    CORA_CHECK(priv_broadcastable(inf_.shape(), factor.shape()), Error::dimension_mismatch, [&] {
        return "times: the shapes " + shape_text(inf_.shape()) + " and " +
               shape_text(factor.shape()) + " do not broadcast";
    });
    const auto [lo, hi] = priv_times(inf_, sup_, factor, factor);
    return {lo, hi};
}

Set Interval::times_impl(const Tensor &factor) const { return times(factor); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
