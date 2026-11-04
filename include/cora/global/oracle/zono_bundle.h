// zono_bundle - lean::ZonoBundle, a zonotope bundle of exact values, computed by CORALean
//
// Syntax:   lean::ZonoBundle bundle(cora_bundle);   lean::ZonoBundle r = bundle.intersect(other);
// r.nominal(); Operations: one method per `zonoBundle.*` op of the oracle, named as CORA.cpp names
// CORA's (`and`
//             is intersect, `cartProd` cart_prod); the ops of CORA's methods that CORA.cpp ports
// Parts:      the oracle's bundle is a list of lean zonotopes, each with its own error box (at
//             dtype nearest they are empty); interval() is the box the oracle proves to enclose it
// Crossing:   nominal() returns the nominal parts of the members as a cora::ZonoBundle, exactly or
//             throws
// See also:   lean/zonotope.h, lean/con_zonotope.h, lean/request.h,
//             cora/cont_set/zono_bundle/zono_bundle.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"
#include "cora/global/oracle/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class ZonoBundle {
  public:
    /// An exact bundle (empty error boxes) in the current dtype.
    ZonoBundle(const cora::ZonoBundle &bundle);
    explicit ZonoBundle(std::vector<Zonotope> zonotopes);

    /// The members, each a lean zonotope with its error box.
    const std::vector<Zonotope> &zonotopes() const noexcept { return Z_; }

    /// The nominal part of every member as a cora::ZonoBundle; throws if a value is not a double.
    cora::ZonoBundle nominal() const;

    // Operations ---------------------------------------------------------------------------------

    ZonoBundle intersect(const ZonoBundle &other) const;
    ZonoBundle cart_prod(const ZonoBundle &other) const;

    /// The enclosure of this bundle and M * this + other (the oracle's `enclose`).
    ZonoBundle enclose(const Tensor &M, const ZonoBundle &other) const;

    /// Every member scaled about its center by factor (n, 1).
    ZonoBundle enlarge(const Tensor &factor) const;

    /// Every member lifted: state i at position dims[i], N = dims.size().
    ZonoBundle lift(int64_t N, const std::vector<int64_t> &dims) const;

    static ZonoBundle empty(int64_t n);

    /// The constrained zonotope of the bundle (`zonoBundle.conZonotope`).
    ConZonotope con_zonotope() const;

    /// The box the oracle proves to enclose the bundle (`zonoBundle.interval`).
    Interval interval() const;

    bool is_bounded() const;

    /// A true `disjoint` proves the boxes around the two bundles are apart (the answer of CORA's
    /// "approx"): the result is certified, true (intersecting) unproved, false proved.
    Certified is_intersecting(const ZonoBundle &other, double tol = 0) const;

    bool isequal(const ZonoBundle &other, double tol = 0) const;

    Json to_json() const;
    static ZonoBundle from_json(const std::string &dtype, const Json &j);

  private:
    std::vector<Zonotope> Z_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
