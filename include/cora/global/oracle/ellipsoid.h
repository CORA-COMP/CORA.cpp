// ellipsoid - lean::Ellipsoid, an ellipsoid of exact values, computed by CORALean
//
// Syntax:     lean::Ellipsoid E(cora_E);   lean::Ellipsoid R = E.cart_prod(p);   R.gather();
// Operations: one method per `ellipsoid.*` op of the oracle, named as CORA.cpp names CORA's
//             (`cartProd` is cart_prod with a point, `isFullDim` is_full_dim, `isBigger` is_bigger)
// Parts:      the oracle's set is {Q, q} and the error box E of the sound float operations; at
//             dtype nearest the box is empty
// Crossing:   nominal() returns Q and q as a cora::Ellipsoid, exactly or throws
// See also:   lean/con_zonotope.h, lean/request.h, cora/cont_set/ellipsoid/ellipsoid.h
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/oracle/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Ellipsoid {
  public:
    /// An exact ellipsoid (empty error box) in the current dtype.
    Ellipsoid(const cora::Ellipsoid &E);

    /// The dtype of the values.
    const std::string &dtype() const noexcept { return dtype_; }

    /// The shape matrix Q and the center q as the oracle returns them; throws if a value is not a
    /// double. The error box is not looked at.
    cora::Ellipsoid nominal() const;

    /// Whether the error box is empty (all zero).
    bool exact() const;

    // Operations ---------------------------------------------------------------------------------

    /// The center q (`ellipsoid.center`), a column.
    Tensor center() const;

    /// Whether Q has full rank (`ellipsoid.isFullDim`, a certificate: true is proved).
    bool is_full_dim() const;

    /// Whether E2 - q2 lies in E - q (`ellipsoid.isBigger`, a certificate: true is proved).
    bool is_bigger(const Ellipsoid &E2) const;

    /// The Cartesian product with the point v (`ellipsoid.cartProd`): exact.
    Ellipsoid cart_prod(const Tensor &v) const;

    Json to_json() const;
    static Ellipsoid from_json(const std::string &dtype, const Json &j);

  private:
    Ellipsoid(std::string dtype, Json json) : dtype_(std::move(dtype)), json_(std::move(json)) {}

    std::string dtype_;
    Json json_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
