// priv_pot_int - the guards that the reachable sets of a location meet, as CORA's location.potInt
//
// Every time-interval set (the time-point sets if there are none) is tested against every guard
// whose target is not a final location, by CORALean's test (Location.potInt): a guard is certified
// disjoint from a set if a row lies strictly beyond it, b < -h(-a) with h the support function
// rounded up; a bundle is disjoint if one of its members is. CORALean does not read the equalities
// of a guard, which makes a hyperplane guard met by every set; CORA.cpp reads an equality as its
// two rows (docs/cora-differences.md), a sound strengthening. A timer that reaches a
// time-triggered hyperplane exactly at a step meets it with the time-point set only: if one or two
// neighbouring time-interval sets meet one hyperplane, the time-point set between them lies on it
// and they do not, that set is taken instead.
//
// Syntax:   GuardHits hits = priv_pot_int(loc, R, final_locations);
// Inputs:   loc - the location
//           R - its reachable sets
//           final_locations - targets that are not followed
// Outputs:  hits - per hit the transition and the set; time_point if the sets are time points
// See also: priv_guard_intersect, priv_location_reach
//
// CORA: hybridDynamics/@location/potInt.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the set may meet the polytope of C: no row lies strictly beyond it, b < -h(-a) (see the
/// docstring of the file for the bundle and the equalities).
bool aux_may_meet(const Polytope::Constraints &C, const Set &S) {
    // the rows of A x <= b, and each equality Ae x = be as the rows Ae x <= be and -Ae x <= -be
    std::vector<std::pair<Tensor, double>> rows;
    const auto add = [&](const Tensor &M, const Tensor &offsets, double sign) {
        const std::vector<double> b = offsets.data();
        for (std::size_t i = 0; i < b.size(); ++i)
            rows.emplace_back(sign * M.slice(-2, int64_t(i), int64_t(i) + 1).matrix_transpose(),
                              sign * b[i]);
    };
    add(C.A, C.b, 1.0);
    if (C.Ae.numel() > 0) {
        add(C.Ae, C.be, 1.0);
        add(C.Ae, C.be, -1.0);
    }
    for (const Zonotope &Z : priv_zonotopes(S))
        for (const auto &[row, offset] : rows)
            if (offset < -Z.support_function(-row).item()) return false;
    return true;
}

/// Whether the guard holds the set, as CORA's contains_(guard, S, 'exact', 1e-10).
bool aux_on_guard(const Polytope &guard, const Set &S) {
    return guard.contains(*S, "exact", numerics::kGuardTimePoint);
}

/// CORA's special case: the time-point set between the hit time-interval sets lies on the
/// hyperplane of the guard and they do not.
bool aux_time_triggered(const Location &loc, const LocationReachSet &R, const GuardHits &hits) {
    const std::size_t count = hits.transitions.size();
    const bool one = count == 1;
    const bool two = count == 2 && hits.transitions[0] == hits.transitions[1] &&
                     hits.sets[1] == hits.sets[0] + 1;
    if (R.time_interval.empty() || !(one || two)) return false;
    // only a polytope can be a hyperplane
    const auto *polytope =
        std::get_if<Polytope>(&loc.transitions()[std::size_t(hits.transitions[0])].guard());
    if (polytope == nullptr) return false;
    const Polytope &guard = *polytope;
    if (!guard.representsa("conHyperplane", numerics::kGuardHyperplane)) return false;
    // the time point at the end of the first set, against the time-interval sets
    if (!aux_on_guard(guard, R.time_point[std::size_t(hits.sets[0] + 1)])) return false;
    return std::none_of(hits.sets.begin(), hits.sets.end(), [&](int64_t j) {
        return aux_on_guard(guard, R.time_interval[std::size_t(j)]);
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The hits ----------------------------------------------------------------------------------------

// The pairs ---------------------------------------------------------------------------------------

GuardHits priv_pot_int(const Location &loc, const LocationReachSet &R,
                       const std::vector<int64_t> &final_locations) {
    GuardHits hits;
    const bool points = R.time_interval.empty();
    const std::vector<Set> &sets = points ? R.time_point : R.time_interval;
    hits.time_point = points;
    // every guard against every set, guard by guard as CORA
    for (std::size_t i = 0; i < loc.transitions().size(); ++i) {
        const Transition &trans = loc.transitions()[i];
        if (std::count(final_locations.begin(), final_locations.end(), trans.target()) > 0)
            continue;
        // a level set meets a set if its constraints may hold on the box of the set
        const auto *level = std::get_if<LevelSet>(&trans.guard());
        const Polytope::Constraints guard =
            level ? Polytope::Constraints{} : std::get<Polytope>(trans.guard()).constraints();
        for (std::size_t j = 0; j < sets.size(); ++j)
            if (level ? level->is_intersecting(*sets[j], numerics::kGuardMeet)
                      : aux_may_meet(guard, sets[j])) {
                hits.transitions.push_back(int64_t(i));
                hits.sets.push_back(int64_t(j));
            }
    }
    if (aux_time_triggered(loc, R, hits)) return {{hits.transitions[0]}, {hits.sets[0] + 1}, true};
    return hits;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
