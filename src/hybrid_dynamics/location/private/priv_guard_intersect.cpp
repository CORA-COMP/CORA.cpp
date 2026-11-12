// priv_guard_intersect - the enclosures of the guard intersections, as CORA's guardIntersect
//
// The hits are grouped by guard, and each group is split where its set indices have a gap, so
// that one group is one crossing of a guard (CORA's order: a split-off part goes to the front).
// A hyperplane guard crossed by time-interval sets keeps only the sets whose flow may point into
// it (checkFlow; a failure keeps them all). The method then encloses the intersection per basis
// of calcBasis; empty intersections are dropped. A level set guard has one method of its own: the
// union of the boxes of the sets, contracted to the guard, intersected with it
// (LevelSet::intersect).
//
// Syntax:   std::vector<GuardIntersection> G = priv_guard_intersect(loc, hits, R, U, method);
//           std::vector<GuardGroup> groups = priv_guard_groups(hits);
// Inputs:   loc - the location
//           hits - which sets meet which guards (priv_pot_int)
//           R - the reachable sets of the location
//           U - the input set in the state space (B U + c)
//           method - alg::ZonoGirard, alg::NondetGuard or alg::ConZonotopeGuard
// Outputs:  G - per crossing the transition, the first and the last set index, and the set
// See also: priv_calc_basis, priv_check_flow, priv_guard_intersect_zono_girard
//
// CORA: hybridDynamics/@location/guardIntersect.m
// CORA: hybridDynamics/@location/guardIntersect_levelSet.m
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

/// The same for a level set guard.
std::vector<Set> aux_check_flow(const Location &loc, const LevelSet &guard, std::vector<Set> P,
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
                               const alg::GuardIntersect &method, bool dense) {
    return std::visit(
        [&](const auto &m) -> std::optional<Set> {
            using M = std::decay_t<decltype(m)>;
            const std::vector<Tensor> B = priv_calc_basis(loc, P, guard, m.enclose, U_state);
            if constexpr (std::is_same_v<M, alg::ZonoGirard>) {
                return priv_guard_intersect_zono_girard(P, guard, B, dense);
            } else if constexpr (std::is_same_v<M, alg::NondetGuard>) {
                return priv_guard_intersect_nondet_guard(P, guard, B, m.box, m.fm_passes);
            } else {
                return priv_guard_intersect_con_zonotope(P, guard, B, m.guard_order, m.box,
                                                         m.fm_passes);
            }
        },
        method);
}

/// CORA's guardIntersect_levelSet: the union of the boxes of the sets, each contracted to the
/// guard (kept as it is where that fails, as CORA's try), as a polynomial zonotope, intersected
/// with the guard; none if that is empty.
std::optional<Set> aux_enclose_level_set(const std::vector<Set> &P, const LevelSet &guard) {
    std::optional<Interval> box;
    for (const Set &S : P) {
        Interval I = S->interval();
        try {
            // domain tightening needs equalities; an empty contraction keeps the box as well
            if (const std::optional<Interval> tight = guard.tighten_domain(I)) I = *tight;
        } catch (const Error &) {
            I = S->interval(); // CORA's try: the box stays as it is
        }
        box = box ? box->unite(I) : I;
    }
    const std::optional<PolyZonotope> R = guard.intersect(PolyZonotope(*box));
    if (!R) return std::nullopt;
    return Set(*R);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The crossings -----------------------------------------------------------------------------------

/// CORA's aux_groupSets and aux_removeGaps: the hits of each guard (in the order of the guard
/// numbers), split at every gap; the first part of a split goes to the front of the list.
std::vector<GuardGroup> priv_guard_groups(const GuardHits &hits) {
    // the guard numbers that were hit, ascending
    std::vector<int64_t> guards = hits.transitions;
    std::sort(guards.begin(), guards.end());
    guards.erase(std::unique(guards.begin(), guards.end()), guards.end());
    std::vector<GuardGroup> groups;
    // the sets of each guard, in the order of the hits
    for (int64_t g : guards) {
        GuardGroup group{g, {}};
        for (std::size_t k = 0; k < hits.sets.size(); ++k)
            if (hits.transitions[k] == g) group.sets.push_back(hits.sets[k]);
        groups.push_back(std::move(group));
    }
    // split at the gaps
    for (std::size_t counter = 0; counter < groups.size(); ++counter) {
        const std::vector<int64_t> sets = groups[counter].sets;
        // the first gap splits the group: the sets after it stay, the ones before go to the front
        for (std::size_t i = 0; i + 1 < sets.size(); ++i) {
            if (sets[i + 1] == sets[i] + 1) continue;
            const int64_t g = groups[counter].transition;
            groups[counter].sets.assign(sets.begin() + std::ptrdiff_t(i) + 1, sets.end());
            groups.insert(groups.begin(),
                          GuardGroup{g, {sets.begin(), sets.begin() + std::ptrdiff_t(i) + 1}});
            break;
        }
    }
    return groups;
}

// The enclosures ----------------------------------------------------------------------------------

std::vector<GuardIntersection> priv_guard_intersect(const Location &loc, const GuardHits &hits,
                                                    const LocationReachSet &R,
                                                    const Zonotope &U_state,
                                                    const alg::GuardIntersect &method,
                                                    bool check_flow, bool dense) {
    std::vector<GuardIntersection> out;
    const std::vector<Set> &sets = hits.time_point ? R.time_point : R.time_interval;
    for (const GuardGroup &group : priv_guard_groups(hits)) {
        const Region &region = loc.transitions()[std::size_t(group.transition)].guard();
        std::vector<Set> P;
        P.reserve(group.sets.size());
        for (int64_t j : group.sets) P.push_back(sets[std::size_t(j)]);
        // a level set is crossed only where the flow points out along its gradient
        if (const auto *level = std::get_if<LevelSet>(&region)) {
            if (check_flow && !hits.time_point) {
                P = aux_check_flow(loc, *level, std::move(P), U_state);
                if (P.empty()) continue;
            }
            // the only method of a level set is the one of Kochdumper et al. (HSCC 2020)
            if (std::optional<Set> S = aux_enclose_level_set(P, *level))
                out.push_back({group.transition, group.sets.front(), group.sets.back(), *S});
            continue;
        }
        const Polytope &guard = std::get<Polytope>(region);
        // a hyperplane is crossed only where the flow points into it
        if (check_flow && !hits.time_point &&
            guard.representsa("conHyperplane", numerics::kGuardHyperplane)) {
            P = aux_check_flow(loc, guard, std::move(P), U_state);
            if (P.empty()) continue;
        }
        std::optional<Set> S = aux_enclose(loc, P, guard, U_state, method, dense);
        if (S) out.push_back({group.transition, group.sets.front(), group.sets.back(), *S});
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
