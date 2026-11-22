// priv_guard_intersect_level_set - the enclosure of a level set guard crossing, as CORA's method
//
// The union of the boxes of the sets, each contracted to the guard (kept as it is where that
// fails, as CORA's try), as a polynomial zonotope, intersected with the guard (LevelSet::intersect)
// [1, Sec. 3.2]. Only the instances with level set guards run this file.
//
// Syntax:   std::optional<Set> S = priv_guard_intersect_level_set(R, guard);
// Inputs:   R - the sets of one crossing
//           guard - the level set guard
// Outputs:  S - the enclosure; none if the intersection is empty
// See also: priv_guard_intersect
// References:
//    [1] N. Kochdumper, M. Althoff. "Reachability analysis for hybrid systems with nonlinear guard
//        sets", HSCC 2020.
//
// CORA: hybridDynamics/@location/guardIntersect_levelSet.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   16-November-2026
// Reviewed:      ---

#include "hybrid_dynamics/location/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::optional<Set> priv_guard_intersect_level_set(const std::vector<Set> &R,
                                                  const LevelSet &guard) {
    std::optional<Interval> box;
    for (const Set &S : R) {
        Interval I = S->interval();
        try {
            // domain tightening needs equalities; an empty contraction keeps the box as well
            if (const std::optional<Interval> tight = guard.tighten_domain(I)) I = *tight;
        } catch (const Error &) {
            I = S->interval(); // CORA's try: the box stays as it is
        }
        box = box ? box->unite(I) : I;
    }
    const std::optional<PolyZonotope> P = guard.intersect(PolyZonotope(*box));
    if (!P) return std::nullopt;
    return Set(*P);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
