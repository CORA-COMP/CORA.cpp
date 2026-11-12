// hybrid_automaton - lean::HybridAutomaton, the reach loop of a hybrid automaton by CORALean
//
// The oracle's op `hybridAutomaton.reach`: CORA's loop of hybridAutomaton/reach with its queue,
// at linear flows, halfspace invariants and bare hyperplane guards, the guard intersection
// zonoGirard in the basis 'box'. Not modelled, so different from CORA: checkFlow, the swap of a
// lone hyperplane hit for the time-point set, and the specifications.
//
// Syntax:     auto branches = lean::HybridAutomaton::reach(locations, 0, R0, 0, t_final, 10, 50);
// Operations: reach
// Outputs:    per stay the location, the producing branch and the time-interval sets
// See also:   lean/location.h, hybrid_dynamics/hybrid_automaton/hybrid_automaton.h
//
// CORA.cpp only: the wrapper of the oracle, CORA has no such class
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/location.h"
#include "cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h"

#include <array>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// A transition: the hyperplane guard, the reset x -> W x + w and the target location.
struct HybridTransition {
    cora::Polytope guard;
    cora::Tensor W, w;
    int64_t target;
};

/// A location: the flow with its input set, the invariant, the transitions and the time step.
struct HybridLocation {
    Flow flow;
    cora::Polytope invariant;
    std::vector<HybridTransition> transitions;
    double time_step;
};

/// A stay of the analysis: its location, the branch that produced it (none for the first), its
/// time-interval sets and the lower and upper end of its start time.
struct HybridBranch {
    int64_t location;
    std::optional<int64_t> parent;
    std::vector<cora::Zonotope> time_interval;
    std::array<double, 2> t_start;
};

struct HybridAutomaton {
    /// The branches in the order of analysis, from R0 in start_loc at t_start up to t_final; the
    /// transitions into final_loc (default: none) are not scanned. check_flow maps onto the
    /// op's field `checkFlow` (sent only when true; the oracle of !927 has none).
    static std::vector<HybridBranch>
    reach(const std::vector<HybridLocation> &locations, int64_t start_loc, const cora::Zonotope &R0,
          double t_start, double t_final, int64_t taylor_terms, int64_t zonotope_order,
          std::optional<int64_t> final_loc = std::nullopt, bool check_flow = false);

    /// The same for a CORA.cpp automaton of linear flows with polytope invariants and guards, the
    /// problem and the algorithm "standard" (its time steps, orders and final location); the
    /// specifications play no part. Fails for what the op does not serve.
    static std::vector<HybridBranch> reach(const cora::HybridAutomaton &HA,
                                           const cora::HybridReachProblem &problem,
                                           const cora::alg::HybridStandard &alg);
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
