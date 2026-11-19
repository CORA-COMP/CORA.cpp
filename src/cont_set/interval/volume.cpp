// volume - the volume of a box, as CORA's interval.volume_
//
// 0 if the box is empty or some width is 0, also next to an infinite one. The widths multiply
// nested to the right, as CORALean's volume, so the roundings agree with it. An interval matrix
// multiplies the entries stacked column by column (CORA's intervalMatrix volume), 0 without rows.
//
// Syntax:   Tensor v = I.volume();
// Inputs:   -
// Outputs:  v - the product of the widths of all entries, (batch...)
// See also: half_width, ball_radius
//
// CORA: contSet/@interval/volume_.m
// CORA: matrixSet/@intervalMatrix/volume.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_vec.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::volume() const {
    detail::Entry entry;
    // an interval matrix is the box of its entries, stacked column by column
    const Interval box = elem_rank_ == 2 ? priv_vec(*this) : *this;
    const Tensor width = box.sup_ - box.inf_;
    const Tensor product = width.prod(-1);
    if (is_empty() || width.size(-1) == 0) return product.zeros_like();
    const Tensor degenerate = width.equal(width.zeros_like()).any(-1);
    return Tensor::where(degenerate, product.zeros_like(), product);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
