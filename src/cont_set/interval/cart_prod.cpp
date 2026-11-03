// cart_prod - the Cartesian product with a box or a point, as CORA's interval.cartProd_
//
// Syntax:   Interval P = I.cart_prod(I2);   Interval P = I.cart_prod(p);
//           Interval P = Interval::cart_prod(p, I);
// Inputs:   I2 - interval;  p - a point, a column (..., m, 1)
// Outputs:  P - the columns stacked (I first), or for rows the rows joined side by side; an empty
//           operand returns the other; nothing rounds
// See also: cat, cart_prod (registry)
//
// CORA: contSet/@interval/cartProd_.m

#include "cont_set/interval/private/priv_join.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The dimension the boxes join along: 1 for two columns, 2 for two rows, else 0.
int aux_join_dim(const Interval &I, const Interval &I2) {
    const Tensor &a = I.inf(), &b = I2.inf();
    if (a.shape().back() == 1 && b.shape().back() == 1 && priv_joinable(a, b, 1)) return 1;
    return a.size(-2) == 1 && b.size(-2) == 1 && priv_joinable(a, b, 2) ? 2 : 0;
}

/// What the check of a point says.
std::string aux_point_message(const Tensor &p) {
    return "cart_prod: the point must be a column (..., m, 1), got " + shape_text(p.shape());
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
        return "cart_prod: the intervals must both be columns or both rows of one batch shape, "
               "got the bounds " +
               shape_text(inf_.shape()) + " and " + shape_text(I2.inf().shape());
    });
    return priv_join({*this, I2}, dim);
}

Interval Interval::cart_prod(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= 2, Error::dimension_mismatch, [&] { return aux_point_message(p); });
    // the point is the degenerate box [p, p]
    return cart_prod(Interval(p, p));
}

Interval Interval::cart_prod(const Tensor &p, const Interval &I) {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= 2, Error::dimension_mismatch, [&] { return aux_point_message(p); });
    // the point is the degenerate box [p, p]
    return Interval(p, p).cart_prod(I);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
