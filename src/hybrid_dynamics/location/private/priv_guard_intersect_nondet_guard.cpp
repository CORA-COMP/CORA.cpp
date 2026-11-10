// priv_guard_intersect_nondet_guard - the guard intersection of uncertain guards (nondetGuard)
//
// Per basis B, the sets are enclosed by the box of their images B'R, united, and mapped back;
// that zonotope is cut by the guard (its rows kept on the factors, ConZonotope::cut), and the cut
// is enclosed by its box in the same basis, from fm_passes Fourier-Motzkin passes (CORALean's
// guardIntersectNondetHFloat; CORA solves a linear program per bound). Several bases give a bundle.
//
// Syntax:   std::optional<Set> S = priv_guard_intersect_nondet_guard(R, guard, B, fm_passes);
// Inputs:   R - the sets that meet the guard
//           guard - the guard (any polytope)
//           B - the bases (n, n)
//           fm_passes - the passes that box a cut; none: CORA's linear programs (exact, slower)
// Outputs:  S - the enclosure (a zonotope, or a bundle for several bases); none if empty
// See also: priv_guard_intersect, priv_calc_basis
//
// CORA: hybridDynamics/@location/guardIntersect_nondetGuard.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "hybrid_dynamics/location/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::optional<Set> priv_guard_intersect_nondet_guard(const std::vector<Set> &R,
                                                     const Polytope &guard,
                                                     const std::vector<Tensor> &B,
                                                     std::optional<int> fm_passes) {
    std::vector<Zonotope> parts;
    for (const Tensor &basis : B) {
        // the union of the boxes of the sets in the basis, back in the state space
        const Tensor Bt = basis.matrix_transpose();
        std::optional<Interval> I;
        for (const Set &S : R) {
            const Interval box = S->linear_map(Bt)->interval();
            if (!box.is_empty()) I = I ? I->unite(box) : box;
        }
        if (!I) return std::nullopt;
        const Zonotope Z = I->zonotope().linear_map(basis);
        // its cut by the guard, boxed in the same basis
        const ConZonotope cz(Z);
        const Interval cut = fm_passes ? cz.cut(guard).linear_map(Bt).interval_tight(*fm_passes)
                                       : cz.intersect(guard).linear_map(Bt).interval();
        if (cut.is_empty()) return std::nullopt; // CORA drops an empty part, and so the set
        parts.push_back(cut.zonotope().linear_map(basis));
    }
    return priv_bundle(std::move(parts));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
