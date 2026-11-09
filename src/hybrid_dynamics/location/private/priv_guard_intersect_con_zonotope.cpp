// priv_guard_intersect_con_zonotope - the guard intersection by constrained zonotopes
//
// Each set is reduced to guard_order (Girard) and intersected with the guard as a constrained
// zonotope, exactly; per basis B the boxes of the images B' of the intersections are united and
// mapped back. Several bases give a zonotope bundle.
//
// Syntax:   std::optional<Set> S = priv_guard_intersect_con_zonotope(R, guard, B, order);
// Inputs:   R - the sets that meet the guard
//           guard - the guard (any polytope)
//           B - the bases (n, n)
//           order - the order of the reduction first (CORA's guardOrder)
// Outputs:  S - the enclosure (a zonotope, or a bundle for several bases); none if empty
// See also: priv_guard_intersect, priv_calc_basis
//
// CORA: hybridDynamics/@location/guardIntersect_conZonotope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "hybrid_dynamics/location/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The set reduced to the order, as a constrained zonotope (a bundle through its conversion).
ConZonotope aux_reduced(const Set &S, int order) {
    if (S.is<Zonotope>()) return ConZonotope(S.as<Zonotope>().reduce(order));
    return S.as<ZonoBundle>().reduce(order).con_zonotope();
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<Set> priv_guard_intersect_con_zonotope(const std::vector<Set> &R,
                                                     const Polytope &guard,
                                                     const std::vector<Tensor> &B,
                                                     int guard_order) {
    // the exact intersections of the reduced sets with the guard
    std::vector<ConZonotope> cut;
    cut.reserve(R.size());
    for (const Set &S : R) cut.push_back(aux_reduced(S, guard_order).intersect(guard));
    std::vector<Zonotope> parts;
    for (const Tensor &basis : B) {
        // the union of their boxes in the basis, back in the state space
        const Tensor Bt = basis.matrix_transpose();
        std::optional<Interval> I;
        for (const ConZonotope &cz : cut) {
            const Interval box = cz.linear_map(Bt).interval();
            if (!box.is_empty()) I = I ? I->unite(box) : box;
        }
        if (!I) return std::nullopt;
        parts.push_back(I->zonotope().linear_map(basis));
    }
    return priv_bundle(std::move(parts));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
