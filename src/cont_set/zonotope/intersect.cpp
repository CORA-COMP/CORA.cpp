// intersect - a zonotope that encloses the intersection of two zonotopes, as CORA's zonotope.and
//
// Syntax:   Zres = Z.intersect(Z2);   Zres = Z.intersect(Z2, "averaging");
// Inputs:   Z2 - zonotope of the same dimension
//           method - "conZonotope" (CORA's default) or "averaging"; two boxes (zonotopes with at
//                    most one nonzero entry per generator, up to eps) are intersected exactly
//                    whatever the method
// Outputs:  Zres - for two boxes, their intersection; for "averaging", the zonotope with the
//           weighted centers and generators w_i G_i / sum(w) (w_i ~ 1 / trace(G_i G_i'));
//           "conZonotope" needs ConZonotope and raises not_supported for other pairs
// See also: enclose, interval
//
// CORA: contSet/@zonotope/and_.m
// CORA: contSet/@zonotope/private/priv_andAveraging.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"
#include "cont_set/zonotope/private/priv.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// The intersection of two boxes given as zonotopes (CORA's interval and_), as a zonotope.
Zonotope aux_intersect_boxes(const Zonotope &Z1, const Zonotope &Z2) {
    const Interval I1 = Z1.interval(), I2 = Z2.interval();
    const Tensor lower = I1.inf().maximum(I2.inf()), upper = I1.sup().minimum(I2.sup());
    if ((lower - upper).detach().max().item() > kEps)
        CORA_FAIL(Error::empty_set,
                  "intersect: the boxes do not intersect and CORA.cpp has no empty set yet");
    return priv_zonotope_of_interval({lower.minimum(upper), lower.maximum(upper)});
}

/// The weighted average of the zonotopes (CORA's priv_andAveraging, 'normGen' in closed form):
/// weights proportional to 1 / trace(G G'), summing to 1.
Zonotope aux_average(const Zonotope &Z1, const Zonotope &Z2) {
    const auto trace = [](const Zonotope &Z) {
        return Z.generators().matmul(Z.generators().matrix_transpose()).trace();
    };
    const Tensor t1 = trace(Z1), t2 = trace(Z2);
    const Tensor inverse_sum = (1.0 / t1) + (1.0 / t2);
    const Tensor w1 = 1.0 / t1.multiply(inverse_sum), w2 = 1.0 / t2.multiply(inverse_sum);
    const Tensor total = w1 + w2;
    const Tensor c = w1.multiply(Z1.center()).add(w2.multiply(Z2.center())).divide(total);
    const Tensor G = Tensor::concat(
        {w1.multiply(Z1.generators()), w2.multiply(Z2.generators())}, -1);
    return {c, (1.0 / total).multiply(G)};
}

/// An enclosure of the intersection of Z1 and Z2: exact for boxes, else by the method.
Zonotope aux_intersect(const Zonotope &Z1, const Zonotope &Z2, const std::string &method) {
    if (priv_is_box(Z1, kEps) && priv_is_box(Z2, kEps)) return aux_intersect_boxes(Z1, Z2);
    if (method == "averaging") return aux_average(Z1, Z2);
    CORA_FAIL(Error::not_supported, "intersect: method 'conZonotope' for zonotopes that are not "
                                    "boxes needs the class ConZonotope, which does not exist yet");
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::intersect(const Zonotope &Z2, const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(Z2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "intersect: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z2.dim());
    });
    CORA_CHECK(method == "conZonotope" || method == "averaging", Error::wrong_value, [&] {
        return "intersect: the method must be 'conZonotope' or 'averaging', got '" + method + "'";
    });
    return aux_intersect(*this, Z2, method);
}

Zonotope Zonotope::intersect(const Interval &I, const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(I.dim() == dim() && I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "intersect: the interval must be a box of the dimension " + std::to_string(dim()) +
               " of the zonotope, got bounds of the shape " + shape_text(I.inf().shape());
    });
    CORA_CHECK(method == "conZonotope" || method == "averaging", Error::wrong_value, [&] {
        return "intersect: the method must be 'conZonotope' or 'averaging', got '" + method + "'";
    });
    return aux_intersect(*this, priv_zonotope_of_interval(I), method);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
