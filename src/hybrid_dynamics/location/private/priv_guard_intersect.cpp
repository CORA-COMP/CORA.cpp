// priv_guard_intersect - the enclosures of the guard intersections, as CORA's guardIntersect
//
// The hits are grouped by guard, and each group is split where its set indices have a gap, so
// that one group is one crossing of a guard (CORA's order: a split-off part goes to the front).
// A hyperplane guard crossed by time-interval sets keeps only the sets whose flow may point into
// it (checkFlow; a failure keeps them all). The method then encloses the intersection per basis
// of calcBasis; empty intersections are dropped.
//
// Syntax:   std::vector<GuardIntersection> G = priv_guard_intersect(loc, hits, R, U, method);
// Inputs:   loc - the location
//           hits - which sets meet which guards (priv_pot_int)
//           R - the reachable sets of the location
//           U - the input set in the state space (B U + c)
//           method - alg::ZonoGirard, alg::NondetGuard or alg::ConZonotopeGuard
// Outputs:  G - per crossing the transition, the first and the last set index, and the set
// See also: priv_calc_basis, priv_check_flow, priv_guard_intersect_zono_girard
//
// CORA: hybridDynamics/@location/guardIntersect.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The crossings -----------------------------------------------------------------------------------

/// One crossing: the transition and the indices of its sets, without gaps.
struct Group {
    int64_t transition;
    std::vector<int64_t> sets;
};

/// CORA's aux_groupSets and aux_removeGaps: the hits of each guard (in the order of the guard
/// numbers), split at every gap; the first part of a split goes to the front of the list.
std::vector<Group> aux_groups(const GuardHits &hits) {
    std::vector<int64_t> guards = hits.transitions;
    std::sort(guards.begin(), guards.end());
    guards.erase(std::unique(guards.begin(), guards.end()), guards.end());
    std::vector<Group> groups;
    for (int64_t g : guards) {
        Group group{g, {}};
        for (std::size_t k = 0; k < hits.sets.size(); ++k)
            if (hits.transitions[k] == g) group.sets.push_back(hits.sets[k]);
        groups.push_back(std::move(group));
    }
    // split at the gaps
    for (std::size_t counter = 0; counter < groups.size(); ++counter) {
        const std::vector<int64_t> sets = groups[counter].sets;
        for (std::size_t i = 0; i + 1 < sets.size(); ++i) {
            if (sets[i + 1] == sets[i] + 1) continue;
            const int64_t g = groups[counter].transition;
            groups[counter].sets.assign(sets.begin() + std::ptrdiff_t(i) + 1, sets.end());
            groups.insert(groups.begin(),
                          Group{g, {sets.begin(), sets.begin() + std::ptrdiff_t(i) + 1}});
            break;
        }
    }
    return groups;
}

// The enclosure -----------------------------------------------------------------------------------

/// CORA's checkFlow in its try: the sets whose flow may point into the guard, all if it fails.
std::vector<Set> aux_check_flow(const Location &loc, const Polytope &guard, std::vector<Set> P,
                                const Zonotope &U_state) {
    try {
        return priv_check_flow(loc, guard, P, U_state);
    } catch (const Error &) {
        return P; // CORA keeps the sets if the check fails
    }
}

/// The enclosure of one crossing by the chosen method.
std::optional<Set> aux_enclose(const Location &loc, const std::vector<Set> &P,
                               const Polytope &guard, const Zonotope &U_state,
                               const alg::GuardIntersect &method) {
    return std::visit(
        [&](const auto &m) -> std::optional<Set> {
            using M = std::decay_t<decltype(m)>;
            const std::vector<Tensor> B = priv_calc_basis(loc, P, guard, m.enclose, U_state);
            if constexpr (std::is_same_v<M, alg::ZonoGirard>) {
                return priv_guard_intersect_zono_girard(P, guard, B);
            } else if constexpr (std::is_same_v<M, alg::NondetGuard>) {
                return priv_guard_intersect_nondet_guard(P, guard, B);
            } else {
                return priv_guard_intersect_con_zonotope(P, guard, B, m.guard_order);
            }
        },
        method);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<GuardIntersection> priv_guard_intersect(const Location &loc, const GuardHits &hits,
                                                    const LocationReachSet &R,
                                                    const Zonotope &U_state,
                                                    const alg::GuardIntersect &method) {
    std::vector<GuardIntersection> out;
    const std::vector<Set> &sets = hits.time_point ? R.time_point : R.time_interval;
    for (const Group &group : aux_groups(hits)) {
        const Polytope &guard = loc.transitions()[std::size_t(group.transition)].guard();
        std::vector<Set> P;
        P.reserve(group.sets.size());
        for (int64_t j : group.sets) P.push_back(sets[std::size_t(j)]);
        // a hyperplane is crossed only where the flow points into it
        if (!hits.time_point && guard.representsa("conHyperplane", numerics::kGuardHyperplane)) {
            P = aux_check_flow(loc, guard, std::move(P), U_state);
            if (P.empty()) continue;
        }
        std::optional<Set> S = aux_enclose(loc, P, guard, U_state, method);
        if (S) out.push_back({group.transition, group.sets.front(), group.sets.back(), *S});
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
