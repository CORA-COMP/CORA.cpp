// priv_guard_intersect_con_zonotope - the guard intersection by constrained zonotopes
//
// Each set is reduced to guard_order (Girard) and cut by the guard (its rows kept on the factors,
// ConZonotope::cut); per basis B the boxes of the images B' of the cuts, from fm_passes Fourier-
// Motzkin passes or the dual certificates of CORA's linear programs (priv_box_cut), are united and
// mapped back (CORALean's guardIntersectPolytopeHFloat). Several bases give a zonotope bundle. A
// basis that is orthogonal only up to rounding widens the member by the box of (I - B B') x over
// the reduced sets (priv_defect; CORALean's orthDefectFloat).
//
// Syntax:   std::optional<Set> S = priv_guard_intersect_con_zonotope(R, guard, B, order, box, k);
// Inputs:   R - the sets that meet the guard
//           guard - the guard (any polytope)
//           B - the bases (n, n)
//           order - the order of the reduction first (CORA's guardOrder)
//           box - how a cut is boxed (alg::CutBox), fm_passes - the passes of fm and both
//           dense - a generator per axis in every box, as CORALean's Interval.zonotope
// Outputs:  S - the enclosure (a zonotope, or a bundle for several bases); none if empty
// See also: priv_guard_intersect, priv_calc_basis
//
// CORA: hybridDynamics/@location/guardIntersect_conZonotope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "hybrid_dynamics/location/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The set reduced to the order (a bundle member by member).
Set aux_reduced(const Set &S, int order) {
    if (S.is<Zonotope>()) return Set(S.as<Zonotope>().reduce(order));
    return Set(S.as<ZonoBundle>().reduce(order));
}

/// A reduced set as a constrained zonotope (a bundle through its conversion).
ConZonotope aux_con_zonotope(const Set &S) {
    if (S.is<Zonotope>()) return ConZonotope(S.as<Zonotope>());
    return S.as<ZonoBundle>().con_zonotope();
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<Set> priv_guard_intersect_con_zonotope(const std::vector<Set> &R,
                                                     const Polytope &guard,
                                                     const std::vector<Tensor> &B, int guard_order,
                                                     alg::CutBox box, int fm_passes, bool dense) {
    // the reduced sets cut by the guard
    std::vector<Set> reduced;
    std::vector<ConZonotope::Cut> cut;
    reduced.reserve(R.size());
    cut.reserve(R.size());
    for (const Set &S : R) {
        reduced.push_back(aux_reduced(S, guard_order));
        cut.push_back(aux_con_zonotope(reduced.back()).cut(guard));
    }
    std::vector<Zonotope> parts;
    for (const Tensor &basis : B) {
        // the union of their boxes in the basis, back in the state space
        const Tensor Bt = basis.matrix_transpose();
        std::optional<Interval> I;
        for (const ConZonotope::Cut &K : cut) {
            const Interval part = priv_box_cut(K.linear_map(Bt), box, fm_passes);
            if (!part.is_empty()) I = I ? I->unite(part) : part;
        }
        if (!I) return std::nullopt;
        Zonotope member = priv_box_zonotope(*I, dense).linear_map(basis);
        if (const auto defect = priv_defect(basis, reduced, dense)) member = member.plus(*defect);
        parts.push_back(std::move(member));
    }
    return priv_bundle(std::move(parts));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
