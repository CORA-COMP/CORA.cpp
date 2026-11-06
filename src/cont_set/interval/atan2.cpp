// atan2 - the four-quadrant arctangent of two boxes, as CORA's interval.atan2
//
// Syntax:   Interval R = I.atan2(I2);
// Inputs:   I2 - interval of the shape of I, the abscissae x (I holds the ordinates y)
// Outputs:  R - the hull of atan2(y, x) at the four corners of the boxes, elementwise. As in CORA
// it
//           is an enclosure only where the box does not meet the branch cut x < 0, y = 0
// See also: atan, Tensor::atan2
//
// CORA: contSet/@interval/atan2.m
// CORA authors: Sebastian Mair
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::atan2(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.inf().shape() == inf_.shape(), Error::dimension_mismatch, [&] {
        return "atan2: the intervals must have one shape, got the bounds " +
               shape_text(inf_.shape()) + " and " + shape_text(I2.inf().shape());
    });
    const Tensor r1 = inf_.atan2(I2.inf()), r2 = inf_.atan2(I2.sup());
    const Tensor r3 = sup_.atan2(I2.inf()), r4 = sup_.atan2(I2.sup());
    // the corner hull is an enclosure only where the box misses the branch cut
    const Tensor zero = inf_.zeros_like();
    const Tensor off_cut = I2.inf().greater(zero).add(inf_.greater(zero)).add(sup_.less(zero));
    // pi rounded up: twice acos(0), widened by a few ulp
    const double kUp = 2.0 * (1.0 + 4 * machine_eps(inf_.dtype()));
    const Tensor pi = zero.acos() * kUp;
    return {Tensor::where(off_cut, r1.minimum(r2).minimum(r3.minimum(r4)), -pi),
            Tensor::where(off_cut, r1.maximum(r2).maximum(r3.maximum(r4)), pi)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
