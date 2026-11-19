// cart_prod - the Cartesian product with a box or a point, as CORA's interval.cartProd_
//
// An empty operand returns the other; nothing rounds.
//
// Syntax:   Interval P = I.cart_prod(I2);   Interval P = I.cart_prod(p);
//           Interval P = Interval::cart_prod(p, I);
// Inputs:   I2 - interval
//           p - a point (..., m)
// Outputs:  P - the coordinates joined (I first), or for rows the rows joined side by side
// See also: cat, cart_prod (registry)
//
// CORA: contSet/@interval/cartProd_.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_join.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The dimension the boxes join along: 1 for two boxes, 2 for two interval matrices of one row,
/// else 0.
int aux_join_dim(const Interval &I, const Interval &I2) {
    if (I.elem_rank() == 1 && I2.elem_rank() == 1 && priv_joinable(I, I2, 1)) return 1;
    const bool rows = I.elem_rank() == 2 && I2.elem_rank() == 2 && I.dim() == 1 && I2.dim() == 1;
    return rows && priv_joinable(I, I2, 2) ? 2 : 0;
}

/// What the check of a point says.
std::string aux_point_message(const Tensor &p) {
    return "cart_prod: the point must be a vector (..., m), got " + shape_text(p.shape());
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::cart_prod(const Interval &I2) const {
    detail::Entry entry;
    // an empty operand is neutral
    if (is_empty()) return I2;
    if (I2.is_empty()) return *this;
    const int dim = aux_join_dim(*this, I2);
    CORA_CHECK(dim != 0, Error::dimension_mismatch, [&] {
        return "cart_prod: the intervals must both be boxes or both rows of one batch shape, "
               "got the bounds " +
               shape_text(inf_.shape()) + " and " + shape_text(I2.inf().shape());
    });
    return priv_join({*this, I2}, dim);
}

Interval Interval::cart_prod(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= 1, Error::dimension_mismatch, [&] { return aux_point_message(p); });
    // the point is the degenerate box [p, p]
    return cart_prod(Interval(p, p));
}

Interval Interval::cart_prod(const Tensor &p, const Interval &I) {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= 1, Error::dimension_mismatch, [&] { return aux_point_message(p); });
    // the point is the degenerate box [p, p]
    return Interval(p, p).cart_prod(I);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
