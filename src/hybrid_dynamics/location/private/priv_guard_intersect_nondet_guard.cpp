// priv_guard_intersect_nondet_guard - the guard intersection of uncertain guards (nondetGuard)
//
// Per basis B, the sets are enclosed by the box of their images B'R, united, and mapped back;
// that zonotope is cut by the guard (its rows kept on the factors, ConZonotope::cut), and the cut
// is enclosed by its box in the same basis (priv_box_cut: Fourier-Motzkin passes, the dual
// certificates of CORA's linear programs, or both; CORALean's guardIntersectNondetHFloat).
// Several bases give a bundle. A basis that is orthogonal only up to rounding widens the
// parallelotope and the member by the box of (I - B B') x (priv_defect; CORALean's
// orthDefectFloat).
//
// Syntax:   std::optional<Set> S = priv_guard_intersect_nondet_guard(R, guard, B, box, k);
// Inputs:   R - the sets that meet the guard
//           guard - the guard (any polytope)
//           B - the bases (n, n)
//           box - how a cut is boxed (alg::CutBox), fm_passes - the passes of fm and both
//           dense - a generator per axis in every box, as CORALean's Interval.zonotope
// Outputs:  S - the enclosure (a zonotope, or a bundle for several bases); none if empty
// See also: priv_guard_intersect, priv_calc_basis
//
// CORA: hybridDynamics/@location/guardIntersect_nondetGuard.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "hybrid_dynamics/location/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::optional<Set> priv_guard_intersect_nondet_guard(const std::vector<Set> &R,
                                                     const Polytope &guard,
                                                     const std::vector<Tensor> &B, alg::CutBox box,
                                                     int fm_passes, bool dense) {
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
        const std::optional<Zonotope> defect = priv_defect(basis, R, dense);
        Zonotope Z = priv_box_zonotope(*I, dense).linear_map(basis);
        if (defect) Z = Z.plus(*defect);
        // its cut by the guard, boxed in the same basis
        const ConZonotope cz(Z);
        const Interval cut = priv_box_cut(cz.cut(guard).linear_map(Bt), box, fm_passes);
        if (cut.is_empty()) return std::nullopt; // CORA drops an empty part, and so the set
        Zonotope member = priv_box_zonotope(cut, dense).linear_map(basis);
        if (defect) member = member.plus(*defect);
        parts.push_back(std::move(member));
    }
    return priv_bundle(std::move(parts));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
