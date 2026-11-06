// cart_prod - the Cartesian product of an ellipsoid with an ellipsoid or a point
//
// One batch shape. With a point, exact: the centers stack and Q is padded with zeros. With an
// ellipsoid, the ellipsoid of the product of the two bounding boxes: each box is widened to
// radius at least 5e-3 in its small coordinates, then Q = k diag(r^2) around the center of the
// box, r the radii and k the number of nonzero radii (CORA gets the same from a solver when a
// radius is zero).
//
// Syntax:   Ellipsoid R = E.cart_prod(S);   Ellipsoid R = E.cart_prod(p);
//           Ellipsoid R = Ellipsoid::cart_prod(p, E);
// Inputs:   S - an ellipsoid, or a point p (n2, 1); as the first operand, the point p (n1, 1)
//           mode - "outer" (default); "inner" and "exact" are Error::not_supported, as in CORA
// Outputs:  R - of dimension n1 + n2, in the order of the operands
// See also: Interval::cart_prod, Zonotope::cart_prod
//
// CORA: contSet/@ellipsoid/cartProd_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/ellipsoid/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Refusals ----------------------------------------------------------------------------------------

/// The refusal of mode: only "outer" is computed.
detail::Refusal aux_mode_refusal(const std::string &mode) {
    if (mode == "outer") return std::nullopt;
    if (mode == "inner" || mode == "exact")
        return std::pair{Error::not_supported,
                         "cart_prod supports only mode = 'outer' for ellipsoids, got '" + mode +
                             "'"};
    return std::pair{Error::wrong_value, "cart_prod: the mode must be 'exact', 'outer' or 'inner'"};
}

/// The refusal of operands of different batch shapes or an empty one, if any.
detail::Refusal aux_operand_refusal(const Ellipsoid &E1, const Ellipsoid &E2) {
    if (priv_is_empty(E1) || priv_is_empty(E2))
        return std::pair{Error::empty_set, std::string("cart_prod: an ellipsoid is empty")};
    const auto batch = [](const Ellipsoid &E) {
        return std::vector<int64_t>(E.Q().shape().begin(), E.Q().shape().end() - 2);
    };
    if (batch(E1) != batch(E2))
        return std::pair{Error::dimension_mismatch,
                         "cart_prod: the ellipsoids must have the same batch shape, got " +
                             shape_text(E1.Q().shape()) + " and " + shape_text(E2.Q().shape())};
    return std::nullopt;
}

/// Why the point is refused, if it is.
detail::Refusal aux_point_refusal(const Tensor &p) {
    if (p.defined() && p.ndim() >= 2 && p.shape().back() == 1) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "cart_prod: the point must be a column (n, 1), got the shape " +
                         (p.defined() ? shape_text(p.shape()) : std::string("()"))};
}


// Products ----------------------------------------------------------------------------------------

/// The point p as an ellipsoid with Q = 0.
Ellipsoid aux_point(const Tensor &p) {
    const int64_t n = p.shape()[p.ndim() - 2];
    return {priv_zeros(p, n, n), p};
}

/// The product of two ellipsoids by one point: centers stack, Q on the block diagonal.
Ellipsoid aux_with_point(const Ellipsoid &E, const Ellipsoid &point, bool point_first) {
    const Ellipsoid &a = point_first ? point : E, &b = point_first ? E : point;
    return {priv_block_diag(a.Q(), b.Q()), Tensor::concat({a.q(), b.q()}, -2)};
}

/// The ellipsoid of a box (CORA's interval-to-ellipsoid conversion, outer): the radii below the
/// minimum are widened to it, then Q = k diag(r^2) with k the number of nonzero radii.
Ellipsoid aux_of_box(const Interval &box) {
    const Tensor rad = box.rad();
    const Tensor small = rad.less(rad.full_like(rad.shape(), numerics::kBoxRadiusMin));
    // a radius of zero stays zero (the factor would be infinite)
    const Tensor positive = rad.greater(rad.zeros_like());
    const Tensor factor = Tensor::where(
        small.multiply(positive),
        numerics::kBoxRadiusMin / Tensor::where(positive, rad, rad.ones_like()), rad.ones_like());
    const Interval widened = box.enlarge(factor);
    const Tensor r = widened.rad();
    const Tensor k = r.greater(r.zeros_like()).sum(-2, true);
    return {r.square().diag().multiply(k), widened.center()};
}

/// The box of an ellipsoid and the box of another side by side.
Interval aux_boxes(const Ellipsoid &E1, const Ellipsoid &E2) {
    const Interval I1 = E1.interval(), I2 = E2.interval();
    return {Tensor::concat({I1.inf(), I2.inf()}, -2), Tensor::concat({I1.sup(), I2.sup()}, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Entries -----------------------------------------------------------------------------------------

Ellipsoid Ellipsoid::cart_prod(const Ellipsoid &S, const std::string &mode) const {
    detail::Entry entry;
    // the mode and the operands are checked before the boxes are formed
    detail::Refusal refusal = aux_mode_refusal(mode);
    if (!refusal) refusal = aux_operand_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_of_box(aux_boxes(*this, S));
}

Ellipsoid Ellipsoid::cart_prod(const Tensor &p, const std::string &mode) const {
    detail::Entry entry;
    detail::Refusal refusal = aux_mode_refusal(mode);
    if (!refusal) refusal = aux_point_refusal(p);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // a point is an ellipsoid without extent
    const Ellipsoid point = aux_point(p);
    refusal = aux_operand_refusal(*this, point);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_with_point(*this, point, false);
}

Ellipsoid Ellipsoid::cart_prod(const Tensor &p, const Ellipsoid &E, const std::string &mode) {
    detail::Entry entry;
    detail::Refusal refusal = aux_mode_refusal(mode);
    if (!refusal) refusal = aux_point_refusal(p);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // the point comes first in the product
    const Ellipsoid point = aux_point(p);
    refusal = aux_operand_refusal(point, E);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_with_point(E, point, true);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
