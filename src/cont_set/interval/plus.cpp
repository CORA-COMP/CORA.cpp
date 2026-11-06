// plus - the Minkowski sum of two intervals, as CORA's interval.plus
//
// Syntax:   Ires = I.plus(I2);   Ires = I + I2;
// Inputs:   I2 - interval of the shape of I
// Outputs:  Ires - the interval of the summed bounds
// See also: linear_map, Zonotope::plus
//
// CORA: contSet/@interval/plus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::plus(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    return {inf_ + I2.inf(), sup_ + I2.sup()};
}

Set Interval::translate_impl(const Tensor &v) const { return Interval(inf_ + v, sup_ + v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
