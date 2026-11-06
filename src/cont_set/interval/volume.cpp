// volume - the volume of a box, as CORA's interval.volume_
//
// 0 if the box is empty or some width is 0, also next to an infinite one. The widths multiply
// nested to the right, as CORALean's volume, so the roundings agree with it.
//
// Syntax:   Tensor v = I.volume();
// Inputs:   -
// Outputs:  v - the product of the widths of all entries, (..., 1, 1)
// See also: rad, radius
//
// CORA: contSet/@interval/volume_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::volume() const {
    detail::Entry entry;
    const Tensor width = sup_ - inf_;
    const Tensor product = width.prod(-1, true).prod(-2, true);
    if (is_empty()) return product.zeros_like();
    const Tensor degenerate = width.equal(width.zeros_like()).any(-1, true).any(-2, true);
    return Tensor::where(degenerate, product.zeros_like(), product);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
