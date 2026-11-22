// evaluate - the image of a set or of points under a linear reset, as CORA's linearReset.evaluate
//
// A zonotope maps to A X + B U + c, a zonotope bundle to the same map of each of its zonotopes;
// the input set enters only through B. Points map one row each.
//
// Syntax:   Set Xplus = reset.evaluate(X, U);   Tensor xplus = reset.evaluate(x, u);
// Inputs:   X - a zonotope or a zonotope bundle of the pre-state dimension
//           U - a zonotope of the input, optional
//           x - points (..., n)
//           u - inputs (..., m) or one input (m), optional
// Outputs:  Xplus - the image set
//           xplus - the image points (..., n+)
// See also: LinearReset, Transition
//
// CORA: hybridDynamics/@linearReset/evaluate.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/hybrid_dynamics/linear_reset/linear_reset.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A X + B U + c of one zonotope (the inputs are valid).
Zonotope aux_image(const LinearReset &reset, const Zonotope &X, const std::optional<Zonotope> &U) {
    Zonotope Y = X.linear_map(reset.A());
    if (reset.B() && U) Y = Y.plus(U->linear_map(*reset.B()));
    if (reset.c()) Y = Y + *reset.c();
    return Y;
}

/// The same for a polynomial zonotope: the map keeps its factors, the input is independent.
PolyZonotope aux_image(const LinearReset &reset, const PolyZonotope &X,
                       const std::optional<Zonotope> &U) {
    PolyZonotope Y = X.linear_map(reset.A());
    if (reset.B() && U) Y = Y.plus(U->linear_map(*reset.B()));
    if (reset.c()) Y = Y.plus(*reset.c());
    return Y;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Set LinearReset::evaluate(const ContSet &X, const std::optional<Zonotope> &U) const {
    detail::Entry entry;
    // a zonotope or a bundle of the pre-state dimension, and an input set of B's columns
    CORA_CHECK(X.class_id() == ClassId::zonotope || X.class_id() == ClassId::zono_bundle ||
                   X.class_id() == ClassId::poly_zonotope,
               Error::not_supported, [&] {
                   return "evaluate: maps a zonotope, a polynomial zonotope or a zonotope bundle, "
                          "got a " +
                          X.class_name();
               });
    CORA_CHECK(X.dim() == pre_state_dim(), Error::dimension_mismatch, [&] {
        return "evaluate: the set has the dimension " + std::to_string(X.dim()) +
               ", the reset maps " + std::to_string(pre_state_dim());
    });
    CORA_CHECK(!U || !B_ || U->dim() == input_dim(), Error::dimension_mismatch, [&] {
        return "evaluate: the input set has the dimension " + std::to_string(U->dim()) +
               ", B has " + std::to_string(input_dim()) + " columns";
    });
    if (X.class_id() == ClassId::zonotope)
        return aux_image(*this, static_cast<const Zonotope &>(X), U);
    if (X.class_id() == ClassId::poly_zonotope)
        return aux_image(*this, static_cast<const PolyZonotope &>(X), U);
    // a bundle maps zonotope by zonotope
    std::vector<Zonotope> parts;
    for (const Zonotope &Z : static_cast<const ZonoBundle &>(X).zonotopes())
        parts.push_back(aux_image(*this, Z, U));
    return ZonoBundle(std::move(parts));
}

Tensor LinearReset::evaluate(const Tensor &x, const std::optional<Tensor> &u) const {
    detail::Entry entry;
    // points as rows (..., n), the inputs as rows (..., m) or one input (m) for all points
    CORA_CHECK(
        x.ndim() >= 1 && x.shape().back() == pre_state_dim(), Error::dimension_mismatch, [&] {
            return "evaluate: the points must be rows (..., " + std::to_string(pre_state_dim()) +
                   "), got the shape " + shape_text(x.shape());
        });
    CORA_CHECK(!u || !B_ || (u->ndim() >= 1 && u->shape().back() == input_dim()),
               Error::dimension_mismatch, [&] {
                   return "evaluate: the inputs must be rows (..., " + std::to_string(input_dim()) +
                          "), got the shape " + shape_text(u->shape());
               });
    // CORA's columns inside: one column per point
    Tensor y = A_.matmul(x.expand_dims(-1)).squeeze(-1);
    if (B_ && u) y = y + B_->matmul(u->expand_dims(-1)).squeeze(-1);
    if (c_) y = y + *c_;
    return y;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
