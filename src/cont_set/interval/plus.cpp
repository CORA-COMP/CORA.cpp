// plus - the Minkowski sum of two intervals, as CORA's interval.plus
//
// Syntax:   Ires = I.plus(I2);   Ires = I + I2;
// Inputs:   I2 - interval of the shape of I
// Outputs:  Ires - the interval of the summed bounds
// See also: linear_map, Zonotope::plus
//
// CORA: contSet/@interval/plus.m
// CORA: matrixSet/@intervalMatrix/plus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::plus(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    return {inf_ + I2.inf(), sup_ + I2.sup(), std::max(elem_rank_, I2.elem_rank_),
            empty_.either(I2.empty_)};
}

// the generic translation takes a column (..., n, 1); a box adds it as a vector
Set Interval::translate_impl(const Tensor &v) const {
    const Tensor t = elem_rank_ == 1 && is_column(v, dim()) ? v.squeeze(-1) : v;
    return with(inf_ + t, sup_ + t);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
