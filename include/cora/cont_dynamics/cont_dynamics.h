// cont_dynamics - what the dynamic systems share: the problems they are asked, the reachable set
//
// A problem is CORA's params (the initial set, the inputs, the time horizon); how it is solved is
// a separate algorithm struct (CORA's options, see algorithms.h). Both are plain structs with
// defaults and a field list, so C++ fills them with designated initializers and Python with
// keyword arguments.
//
// Syntax:   ReachSet R = sys.reach({.initial_set = X0, .t_final = 2.0}, alg::Standard{.time_step =
// 0.1});
//           auto x = sys.simulate({.initial_state = x0, .t_final = 2.0}, alg::Exact{.time_step =
//           0.1});
// See also: cont_dynamics/algorithms.h, cont_dynamics/linear_sys/linear_sys.h
//
// CORA: contDynamics/@contDynamics/contDynamics.m
// CORA: global/classes/@reachSet/reachSet.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/options.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The reachable sets: time_interval[k] over [k*time_step, (k+1)*time_step], time_point[k] at
/// k*time_step (CORA's reachSet, which stores these times with every set). The algorithms set
/// `time_step`; 0 means it is unknown.
struct ReachSet {
    std::vector<Zonotope> time_interval, time_point;
    double time_step = 0;
};

/// What reach and verify are asked (CORA's params).
struct ReachProblem {
    std::optional<Zonotope> initial_set =
        std::nullopt; ///< the initial set X0 (CORA's R0), required
    std::optional<Zonotope> input_set =
        std::nullopt;   ///< the set U of the input u(t) (CORA's U); none: no input
    double t_final = 0; ///< the time horizon, positive (CORA's tFinal)

    static constexpr auto fields() {
        return std::make_tuple(field("initial_set", &ReachProblem::initial_set),
                               field("input_set", &ReachProblem::input_set),
                               field("t_final", &ReachProblem::t_final));
    }
};

/// What simulate is asked (CORA's params of simulate).
struct SimulationProblem {
    Tensor initial_state; ///< the start points as columns (..., n, N), required (CORA's x0)
    std::optional<Tensor> input =
        std::nullopt;   ///< a constant input per trajectory (m, N), or one (m, 1) (CORA's u)
    double t_final = 0; ///< the time horizon, positive (CORA's tFinal)

    static constexpr auto fields() {
        return std::make_tuple(field("initial_state", &SimulationProblem::initial_state),
                               field("input", &SimulationProblem::input),
                               field("t_final", &SimulationProblem::t_final));
    }
};

/// What simulate_random is asked: trajectories from random points of sets (CORA's params).
struct RandomSimulationProblem {
    std::optional<Set> initial_set =
        std::nullopt; ///< the set of the start points (CORA's R0), required
    std::optional<Set> input_set =
        std::nullopt;   ///< the set of the constant inputs (CORA's U); none: no input
    double t_final = 0; ///< the time horizon, positive (CORA's tFinal)

    static constexpr auto fields() {
        return std::make_tuple(field("initial_set", &RandomSimulationProblem::initial_set),
                               field("input_set", &RandomSimulationProblem::input_set),
                               field("t_final", &RandomSimulationProblem::t_final));
    }
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
