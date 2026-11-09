// hybrid_automaton - a hybrid automaton of linear locations, as CORA's hybridAutomaton
//
// The locations are numbered from 0 in the order given; a transition names its target by that
// number. reach computes the reachable sets of every location the initial set reaches until
// t_final, one branch per visit of a location, and checks the specifications on the way.
//
// Syntax:     HybridAutomaton HA("rendezvous", {loc1, loc2, loc3});
//             HybridReachSet R = HA.reach(problem, alg::HybridStandard{...}, specs);
// Operations: reach (reach.cpp); the constructor checks its input (hybrid_automaton.cpp)
// See also:   cora/hybrid_dynamics/location/location.h, cora/cont_dynamics/algorithms.h
//
// CORA: hybridDynamics/@hybridAutomaton/hybridAutomaton.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/hybrid_dynamics/location/location.h"
#include "cora/specification/specification.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// What reach of a hybrid automaton is asked (CORA's params).
struct HybridReachProblem {
    std::optional<Zonotope> initial_set = std::nullopt; ///< the initial set (CORA's R0), required
    int64_t start_location = 0; ///< the location of the initial set, 0-based (CORA's startLoc)
    double t_start = 0;         ///< the start time (CORA's tStart)
    double t_final = 0;         ///< the time horizon, after t_start (CORA's tFinal)
    /// The input sets of the flows: none (no input), one for every location, or one per location
    /// (CORA's U and U{i}), each a zonotope of the input space of its flow's B.
    std::vector<Zonotope> input_set = {};
    /// Locations where the analysis stops: jumps into them are not followed (CORA's finalLoc).
    std::vector<int64_t> final_locations = {};

    static constexpr auto fields() {
        return std::make_tuple(field("initial_set", &HybridReachProblem::initial_set),
                               field("start_location", &HybridReachProblem::start_location),
                               field("t_start", &HybridReachProblem::t_start),
                               field("t_final", &HybridReachProblem::t_final),
                               field("input_set", &HybridReachProblem::input_set),
                               field("final_locations", &HybridReachProblem::final_locations));
    }
};

/// A specification that holds in some locations (CORA's specification with a location).
struct HybridSpecification {
    Specification spec;
    std::vector<int64_t> locations = {}; ///< where it is active, 0-based; empty: everywhere
};

/// The time [lower, upper] of a set; a jump makes the times uncertain.
using TimeRange = std::array<double, 2>;

/// The reachable sets of one visit of a location (CORA's reachSet with loc and parent): the sets
/// at the step times and over the steps, each a zonotope or a zonotope bundle, with their times.
struct LocationReachSet {
    int64_t location = 0; ///< the location, 0-based
    int64_t parent = -1;  ///< the branch whose jump started this one; -1 for the first
    std::vector<Set> time_point, time_interval;
    std::vector<TimeRange> time_point_times, time_interval_times;
};

/// What reach returns (CORA's R and res): the branches in the order they were computed, and
/// whether every specification held; reach stops at the first violation.
struct HybridReachSet {
    std::vector<LocationReachSet> branches;
    bool verified = true;
};

class HybridAutomaton {
  public:
    /// The automaton of the locations, which must all have the same number of states; every
    /// target names one of them.
    HybridAutomaton(std::string name, std::vector<Location> locations);

    const std::string &name() const noexcept { return name_; }
    const std::vector<Location> &locations() const noexcept { return locations_; }

    /// The number of states.
    int64_t dim() const { return locations_.front().dim(); }

    /// The reachable sets from problem.initial_set in problem.start_location up to t_final, the
    /// specifications checked on every set of a location where they are active.
    HybridReachSet reach(const HybridReachProblem &problem, const alg::HybridReach &alg,
                         const std::vector<HybridSpecification> &specs = {}) const;

  private:
    std::string name_;
    std::vector<Location> locations_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
